import torch
from torch import no_grad
from transformers import AutoTokenizer, AutoModel
from src.VT5 import VT5ForConditionalGeneration
from src ._modules import CustomT5Config
from src._model_utils import mean_pooling
from src.utils import flatten, concatenate_patches
from typing import Tuple, Any, List
from PIL import Image
import numpy as np
from time import time

class RAGVT5(torch.nn.Module):
	def __init__(self, config: dict):
		super(RAGVT5, self).__init__()

		# Load config
		self.model_path = config.get("model_weights", "rubentito/vt5-base-spdocvqa")
		print(f"Loading model from {self.model_path}")
		self.page_retrieval = config["page_retrieval"].lower() if "page_retrieval" in config else None
		self.max_source_length = config.get("max_source_length", 512)
		self.device = config.get("device", "cuda")
		self.embed_model = config.get("embed_model", "VT5")
		self.add_sep_token = config.get("add_sep_token", False)

		t5_config = CustomT5Config.from_pretrained(self.model_path, ignore_mismatched_sizes=True)
		t5_config.visual_module_config = config.get("visual_module", {})

		# Load generator
		self.generator = VT5ForConditionalGeneration.from_pretrained(
			self.model_path, config=t5_config, ignore_mismatched_sizes=True
		)
		self.generator.load_config(config)

		if self.add_sep_token:
			# Add the chunk separator token to the tokenizer if not already present
			token_id = self.generator.tokenizer.encode("<sep>", add_special_tokens=False)
			if not(len(token_id) == 1 and token_id[0] != self.generator.tokenizer.unk_token_id):
				print("Adding <sep> token to the tokenizer. This model should now be fine-tuned.")
				self.generator.tokenizer.add_tokens(["<sep>"])
				self.generator.language_backbone.resize_token_embeddings(len(self.generator.tokenizer))

		# Load embedding model
		if self.embed_model == "VT5":
			self.embedding_dim = 768
		elif self.embed_model == "BGE":
			self.bge_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
			self.bge_model = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5')
			self.bge_model.eval()
			self.embedding_dim = 384

	def to(self, device: Any):
		self.device = device
		self.generator.to(device)
		if self.embed_model == "BGE":
			self.bge_model.to(device)

	def eval(self):
		self.generator.eval()

	def train(self):
		self.generator.train()

	@no_grad()
	def embed(self, text: List[str]) -> torch.Tensor:
		if not text:
			return torch.empty(0, self.embedding_dim).to(self.device)
		if self.embed_model == "VT5":
			input_ids, attention_mask = self.generator.tokenizer(
				text,
				return_tensors="pt",
				padding=True,
				truncation=True
			).values()
			input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
			text_tokens_embeddings = self.generator.language_backbone.shared(input_ids)
			text_embeddings = mean_pooling(text_tokens_embeddings, attention_mask)
		elif self.embed_model == "BGE":
			encoded_input = self.bge_tokenizer(
				text,
				return_tensors="pt",
				padding=True,
				truncation=True
			)
			encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
			text_tokens_embeddings = self.bge_model(**encoded_input)[0][:,0]
			text_embeddings = torch.nn.functional.normalize(text_tokens_embeddings, p=2, dim=-1)
		return text_embeddings

	def get_similarities(
			self,
			text_embeddings: List[torch.Tensor], # (bs, n_chunks, hidden_size)
			question_embeddings: torch.Tensor # (bs, hidden_size)
	) -> torch.Tensor: # (bs, n_chunks)
		similarities = []
		bs = len(text_embeddings)

		for i in range(bs):
			text_embeds_i = text_embeddings[i] # (n_chunks_i, hidden_size)
			question_embed_i = question_embeddings[i] # (hidden_size,)
			
			norms_text = torch.norm(text_embeds_i, dim=-1) # (n_chunks_i,)
			norm_question = torch.norm(question_embed_i) # float
			dot_products = torch.matmul(text_embeds_i, question_embed_i) # (n_chunks_i,)
			similarity = dot_products / (norms_text * norm_question + 1e-8) # (n_chunks_i,)
			
			similarities.append(similarity)
		
		return similarities

	def get_chunks(
			self,
			words: list, # (bs, n_pages, n_words)
			boxes: list, # (bs, n_pages, n_words, 4)
			chunk_size: int = 30,
			overlap: int = 0,
			return_words: bool = False
	) -> Tuple[list, list ,list, list, list]: # (bs, n_chunks), (bs, n_chunks, 4), (bs, n_chunks),
											# (bs, n_chunks, n_words), (bs, n_chunks, n_words, 4)
		"""
		Converts words and boxes to chunks
			:param words: list of words
			:param boxes: list of boxes
			:param chunk_size: size of the chunk in words
			:param overlap: overlap of words between chunks
			:param return_words: if True, return also the ocr and boxes for individual words
			:return: text chunks, box chunks, corresponding page indices,
					words text chunks, words box chunks
		"""
		assert chunk_size > 1, "chunk_size should be a non-negative non-zero integer."
		assert overlap >= 0, "overlap should be a non-negative integer."
		assert overlap < chunk_size, "overlap should be less than chunk_size."

		text_chunks = []
		boxes_chunks = []
		page_indices = []
		words_text_chunks = []
		words_boxes_chunks = []
		for b, (batch_words, batch_boxes) in enumerate(zip(words, boxes)):
			batch_text_chunks = []
			batch_box_chunks = []
			batch_page_indices = []
			batch_words_text_chunks = []
			batch_words_box_chunks = []
			for p, (page_words, page_boxes) in enumerate(zip(batch_words, batch_boxes)):
				if self.page_retrieval == "oracle":
					# If oracle, take the whole page as a chunk
					batch_text_chunks.append(" ".join(page_words))
					batch_box_chunks.append([0, 0, 1, 1])
					batch_page_indices.append(p)
					if return_words:
						batch_words_text_chunks.append(page_words)
						batch_words_box_chunks.append(page_boxes)
				else:
					# Else, split the page words and boxes into chunks
					for i in range(0, len(page_words), chunk_size - overlap):
						chunk_words = page_words[i:i + chunk_size]
						chunk_text = " ".join(chunk_words)
						chunk_boxes = page_boxes[i:i + chunk_size]
						batch_text_chunks.append(chunk_text)
						# Find box of the chunk
						min_x = min([box[0] for box in chunk_boxes])
						min_y = min([box[1] for box in chunk_boxes])
						max_x = max([box[2] for box in chunk_boxes])
						max_y = max([box[3] for box in chunk_boxes])
						batch_box_chunks.append([min_x, min_y, max_x, max_y])
						batch_page_indices.append(p)
						if return_words:
							batch_words_text_chunks.append(chunk_words)
							batch_words_box_chunks.append(chunk_boxes)
			text_chunks.append(batch_text_chunks)
			boxes_chunks.append(batch_box_chunks)
			page_indices.append(batch_page_indices)
			words_text_chunks.append(batch_words_text_chunks)
			words_boxes_chunks.append(batch_words_box_chunks)
		return text_chunks, boxes_chunks, page_indices, words_text_chunks, words_boxes_chunks

	@no_grad()
	def retrieve(
			self,
			batch: dict,
			k: int = 5,
			chunk_size: int = 30,
			overlap: int = 0,
			return_words: bool = False,
			include_surroundings: int = 0
	) -> Tuple[list, list, list, list, list, list]: # (bs, k), (bs, k, 4), (bs, k), (bs, k, h, w, 3),
													# (bs, k, n_words), (bs, k, n_words, 4)
		"""
		Retrieve top k chunks and corresponding image patches
			:param batch: input batch with "question", "words", "boxes" and "images"
			:param k: number of chunks to retrieve
			:param chunk_size: size of the chunk in words
			:param overlap: overlap of words between chunks
			:param return_words: if True, return also the ocr and boxes for individual words
			:param include_surroundings: number of words to include before and after each chunk
			:return: top k chunks, top k boxes, top k image patches, top k page indices,
					top k words text chunks, top k words box chunks
		"""
		# Should take [question, pages ocr, pages boxes, pages images]
		# and return [question, chunks ocr, chunks boxes, chunks image patches]
		questions = batch["questions"] # (bs, )
		words = batch["words"] # (bs, n_pages, n_words)
		boxes = batch["boxes"] # (bs, n_pages, n_words, 4)
		images = batch["images"] # (bs, n_pages) PIL images
		bs = len(questions)

		# Get chunks
		text_chunks, box_chunks, page_indices, words_text_chunks, words_box_chunks = \
			self.get_chunks(words, boxes, chunk_size, overlap, return_words)

		# Get text embeddings
		text_embeddings = [] # (bs, n_chunks, hidden_size)
		for text_batch in text_chunks:
			text_embeddings.append(self.embed(text_batch))

		# Get question embeddings
		question_embeddings = self.embed(questions) # (bs, hidden_size)

		# Compute similarity
		similarities = self.get_similarities(text_embeddings, question_embeddings)

		# Get top k chunks and boxes
		top_k_text = []  # (bs, k)
		top_k_boxes = []  # (bs, k, 4)
		top_k_page_indices = []  # (bs, k)
		top_k_words_text = []  # (bs, k, n_words)
		top_k_words_boxes = []  # (bs, k, n_words, 4)

		for b in range(bs):
			k_min = min(k, len(similarities[b]))
			top_k = torch.topk(similarities[b], k=k_min, dim=-1).indices

			# these do not contain surrounding words, just the retrieved chunks
			top_k_text.append([text_chunks[b][i] for i in top_k])
			top_k_boxes.append([box_chunks[b][i] for i in top_k])
			top_k_page_indices.append([page_indices[b][i] for i in top_k])

			# If return words (for generator), also include surrounding words
			if return_words:
				# Build per-page word lists and mappings
				page_words_text = {}             # page_idx -> list of all words on that page
				page_words_boxes = {}            # page_idx -> list of all boxes on that page
				chunk_word_positions = {}        # page_idx -> {chunk_idx: (start_pos, end_pos)}
				included_word_indices = {}       # page_idx -> set of word indices already included

				total_chunks = len(text_chunks[b])

				# Build word lists and chunk positions per page
				for i in range(total_chunks):
					page_idx = page_indices[b][i]
					if page_idx not in page_words_text:
						page_words_text[page_idx] = []
						page_words_boxes[page_idx] = []
						chunk_word_positions[page_idx] = {}
						included_word_indices[page_idx] = set()

					words_in_chunk = words_text_chunks[b][i]
					boxes_in_chunk = words_box_chunks[b][i]
					num_words_in_chunk = len(words_in_chunk)

					# Record the start and end positions of the chunk in the page word list
					start_pos = len(page_words_text[page_idx])
					end_pos = start_pos + num_words_in_chunk

					page_words_text[page_idx].extend(words_in_chunk)
					page_words_boxes[page_idx].extend(boxes_in_chunk)
					chunk_word_positions[page_idx][i] = (start_pos, end_pos)

				# Collect words with surroundings for each top-k chunk
				batch_top_k_words_text = []
				batch_top_k_words_boxes = []

				for i in top_k:
					i = i.item()
					page_idx = page_indices[b][i]
					(start_pos, end_pos) = chunk_word_positions[page_idx][i]

					# Determine the range of word indices to include
					surround_start = max(0, start_pos - include_surroundings)
					surround_end = min(len(page_words_text[page_idx]), end_pos + include_surroundings)

					# Collect word indices in the specified range
					word_indices = range(surround_start, surround_end)

					# Exclude word indices already included
					new_word_indices = [
						idx for idx in word_indices if idx not in included_word_indices[page_idx]
					]

					# Update included word indices
					included_word_indices[page_idx].update(new_word_indices)

					# Collect words and boxes
					words_text = [page_words_text[page_idx][idx] for idx in new_word_indices]
					words_boxes = [page_words_boxes[page_idx][idx] for idx in new_word_indices]

					# Append to batch results
					batch_top_k_words_text.append(words_text)
					batch_top_k_words_boxes.append(words_boxes)

				# Append batch data to the final results
				top_k_words_text.append(batch_top_k_words_text)
				top_k_words_boxes.append(batch_top_k_words_boxes)

		# Get image patches
		top_k_patches = [] # (bs, k, h, w, 3)
		for b in range(bs):
			batch_patches = []
			for i, page_idx in enumerate(top_k_page_indices[b]):
				page: Image.Image = images[b][page_idx] # (H, W, 3)
				box: np.ndarray = top_k_boxes[b][i].copy() # (4,)
				# transform to absolute coordinates
				box[0] = int(box[0] * page.width)
				box[1] = int(box[1] * page.height)
				box[2] = int(box[2] * page.width)
				box[3] = int(box[3] * page.height)
				patch = page.crop(box) # (h, w, 3)
				batch_patches.append(patch)
			top_k_patches.append(batch_patches)

		if self.page_retrieval == "oracle":
			top_k_page_indices = [[batch["answer_page_idx"][b]] for b in range(bs)]

		return top_k_text, top_k_boxes, top_k_patches, top_k_page_indices, top_k_words_text, top_k_words_boxes

	def forward(
			self,
			batch: dict,
			return_pred_answer: bool = True,
			return_retrieval: bool = False,
			chunk_num: int = 5,
			chunk_size: int = 30,
			overlap: int = 0,
			include_surroundings: int = 0
	) -> tuple:
		# Retrieve top k chunks and corresponding image patches
		start_time = time()
		top_k_text, top_k_boxes, top_k_patches, top_k_page_indices, top_k_words_text, top_k_words_boxes = \
			self.retrieve(batch, chunk_num, chunk_size, overlap, True, include_surroundings)
		retrieval_time = time() - start_time
		bs = len(top_k_text)
		# Generate
		start_time = time()
		new_batch = {}
		if self.page_retrieval in ["oracle", "concat"]:
			# Concatenate all the top k chunks
			new_batch["questions"] = batch["questions"].copy() # (bs,)
			new_batch["words"] = [flatten(b, self.add_sep_token) for b in top_k_words_text]  # (bs, k * n_words)
			new_batch["boxes"] = [flatten(b, self.add_sep_token) for b in top_k_words_boxes]  # (bs, k * n_words, 4)
			new_batch["images"] = [concatenate_patches(b, mode="grid") for b in top_k_patches]  # (bs, h, w, 3)
			new_batch["answers"] = batch["answers"].copy()  # (bs, n_answers)
			result = self.generator(new_batch, return_pred_answer=return_pred_answer)  # (4, bs)
		elif self.page_retrieval == "maxconf":
			# Generate for each top k chunk and take the answer with highest confidence
			results = []  # (bs, 4, k)
			max_confidence_indices = []
			for b in range(bs):  # iterate over batch
				words, boxes, patches = [], [], []
				for i in range(len(top_k_words_text[b])):
					if top_k_words_text[b][i] != []:
						words.append(top_k_words_text[b][i])
						boxes.append(top_k_words_boxes[b][i])
						patches.append(top_k_patches[b][i])
				if len(words) == 0:
					# If there are no words, append None and continue
					results.append(None)
					max_confidence_indices.append(None)
					continue
				new_batch["questions"] = [batch["questions"][b]] * len(words)  # (k,)
				new_batch["words"] = words  # (k, n_words)
				new_batch["boxes"] = boxes  # (k, n_words, 4)
				new_batch["images"] = patches  # (k, h, w, 3)
				new_batch["answers"] = [batch["answers"][b]] * len(words)
				# This treats k as batch size
				result = self.generator(new_batch, return_pred_answer=return_pred_answer)
				results.append(result)
				# Get confidence scores
				confidences = torch.tensor(result[3])  # (k,)
				max_confidence_index = torch.argmax(confidences).item()
				max_confidence_indices.append(max_confidence_index)
			# Get the answer with highest confidence
			final_results = [[] for _ in range(4)]  # [[], [], [], []]
			for b in range(bs):
				result = results[b]
				if result is None:
					for i in range(4):
						final_results[i].append(None)
					continue
				conf_idx = max_confidence_indices[b]
				for i in range(4):
					if result[i] is not None:
						final_results[i].append(result[i][conf_idx])
					else:
						final_results[i].append(None)
			result = tuple(final_results)
		generation_time = time() - start_time

		if return_retrieval:
			# For visualization
			retrieval = {
				"text": top_k_text,
				"boxes": top_k_boxes,
				"patches": top_k_patches,
				"page_indices": top_k_page_indices,
				"words_text": top_k_words_text,
				"words_boxes": top_k_words_boxes,
				"retrieval_time": retrieval_time,
				"generation_time": generation_time
			}
			if self.page_retrieval in ["oracle", "concat"]:
				retrieval.update({
					"input_words": new_batch["words"],
					"input_boxes": new_batch["boxes"],
					"input_patches": new_batch["images"]
				})
			elif self.page_retrieval == "maxconf":
				retrieval.update({
					"max_confidence_indices": max_confidence_indices,
					"input_words": [flatten(b) for b in top_k_words_text],
					"input_boxes": [flatten(b) for b in top_k_words_boxes],
					"input_patches": [concatenate_patches(b, mode="grid") for b in top_k_patches]
				})
		else:
			retrieval = None
		return *result, retrieval

	@torch.no_grad
	def inference(
			self,
			batch: dict,
			return_retrieval: bool=False,
			chunk_num: int=5,
			chunk_size: int=30,
			overlap: int=0,
			include_surroundings: int=0,
	):
		self.eval()
		return self.forward(
			batch,
			return_retrieval=return_retrieval,
			chunk_num=chunk_num,
			chunk_size=chunk_size,
			overlap=overlap,
			include_surroundings=include_surroundings
		)
