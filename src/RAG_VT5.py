import torch
from torch import no_grad
from sentence_transformers import SentenceTransformer
from src.VT5 import VT5ForConditionalGeneration
from src.LayoutModel import LayoutModel, layout_map
from src ._modules import CustomT5Config
from src._model_utils import mean_pooling
from src.utils import flatten, concatenate_patches
from typing import Tuple, Any, List
from PIL import Image
import numpy as np
from time import time
from collections import Counter

class RAGVT5(torch.nn.Module):
	def __init__(self, config: dict):
		super(RAGVT5, self).__init__()

		# Load config
		self.model_path = config.get("model_weights", "rubentito/vt5-base-spdocvqa")
		self.embed_path = config.get("embed_weights", None)
		self.layout_model_weights = config.get("layout_model_weights", None)
		self.layout_bs = config.get("layout_batch_size", 1)
		print(f"Loading model from {self.model_path}")
		print(f"Loading embedding model from {self.embed_path}")
		print(f"Loading layout model from {self.layout_model_weights}")
		self.page_retrieval = config["page_retrieval"].lower() if "page_retrieval" in config else None
		self.max_source_length = config.get("max_source_length", 512)
		self.device = config.get("device", "cuda")
		self.embed_model = config.get("embed_model", "VT5")
		self.add_sep_token = config.get("add_sep_token", False)
		self.cache_dir = config.get("cache_dir", None)
		print(f"Using {self.cache_dir} as cache folder")
		self.use_layout_labels = config.get("use_layout_labels", False)
		self.n_stats_examples = 5

		t5_config = CustomT5Config.from_pretrained(self.model_path, ignore_mismatched_sizes=True, cache_dir=self.cache_dir)
		t5_config.visual_module_config = config.get("visual_module", {})

		# Load generator
		self.generator = VT5ForConditionalGeneration.from_pretrained(
			self.model_path, config=t5_config, ignore_mismatched_sizes=True, cache_dir=self.cache_dir
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
		else:
			if self.embed_model == "BGE":
				if self.embed_path is None:
					self.embed_path = "BAAI/bge-small-en-v1.5"
				self.embedding_dim = 384
			elif self.embed_model == "BGE-M3":
				self.embed_path = "BAAI/bge-m3"
				self.embedding_dim = 1024
			elif self.embed_model == "BGE-reranker":
				self.embed_path = "BAAI/bge-reranker-v2-m3"
				self.embedding_dim = 1024
			self.bge_model = SentenceTransformer(self.embed_path, cache_folder=self.cache_dir)

		# Load layout model
		if self.layout_model_weights is not None:
			self.layout_model = LayoutModel(config)
		else:
			self.layout_model = None

	def to(self, device: Any):
		self.device = device
		self.generator.to(device)
		if self.embed_model != "VT5":
			self.bge_model.to(device)

	def eval(self):
		self.generator.eval()

	def train(self):
		self.generator.train()

	def stat_add_example(self, dct: dict, key: Any, example: Any):
		"""
		Add an example value to a list in a dictionary.
		Used for retrieval stats examples.
		"""
		if key not in dct:
			dct[key] = []
		if len(dct[key]) < self.n_stats_examples:
			dct[key].append(example)

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
		else:
			text_embeddings = self.bge_model.encode(text, convert_to_tensor=True)
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
			layout_info: list = None, # (bs, n_pages)
			chunk_size: int = 30,
			chunk_size_tol: float = 0.15,
			overlap: int = 0,
			return_words: bool = False,
			**kwargs
	) -> Tuple[list, list ,list, list, list, list, dict]: # (bs, n_chunks), (bs, n_chunks, 4), (bs, n_chunks),
									# (bs, n_chunks, n_words), (bs, n_chunks, n_words, 4), (bs, n_chunks), dict
		"""
		Converts words and boxes to chunks
			:param words: list of words
			:param boxes: list of boxes
			:param layout_info: list of layout information
			:param chunk_size: size of the chunk in words
			:param overlap: overlap of words between chunks
			:param return_words: if True, return also the ocr and boxes for individual words
			:return: text chunks, box chunks, corresponding page indices,
					words text chunks, words box chunks
		"""
		assert chunk_size > 1, "chunk_size should be a non-negative non-zero integer."
		assert 0 <= chunk_size_tol <= 1, "chunk_size_tol should be a float between 0 and 1."
		assert overlap >= 0, "overlap should be a non-negative integer."
		assert overlap < chunk_size, "overlap should be less than chunk_size."
		bs = len(words)
		question_id = kwargs.get("question_id", None)

		# Extract layout_boxes and layout_labels from layout_info
		if layout_info != [[]]:
			layout_boxes = [[layout_info[b][p]["boxes"] for p in range(len(layout_info[b]))] for b in range(bs)] # (bs, n_pages, n_boxes, 4)
			layout_labels = [[layout_info[b][p]["labels"] for p in range(len(layout_info[b]))] for b in range(bs)] # (bs, n_pages, n_boxes)
		else:
			layout_boxes = None
			layout_labels = None

		text_chunks = [] # (bs, n_chunks)
		boxes_chunks = [] # (bs, n_chunks, 4)
		page_indices = [] # (bs, n_chunks)
		words_text_chunks = [] # (bs, n_chunks, n_words)
		words_boxes_chunks = [] # (bs, n_chunks, n_words, 4)
		layout_labels_chunks = [] # (bs, n_chunks)
		stats = {
			"chunk_size_dist": Counter(),
			"n_chunks_per_layout_dist": Counter(),
			"n_chunks_per_page_dist": Counter(),
			"n_chunks_per_doc_dist": Counter()
		}
		stats_examples = {key: {} for key in stats}

		def make_chunks(_words, _boxes, _p, words_lst, boxes_lst, p_lst) -> int:
			prev_chunk_size = 0
			n_chunks = 0
			for i in range(0, len(_words), chunk_size - overlap):
				chunk_words = _words[i:i + chunk_size]
				chunk_boxes = _boxes[i:i + chunk_size]
				this_chunk_size = len(chunk_words)
				if (
						i > 0 and 
						_p == p_lst[-1] and 
						prev_chunk_size + (this_chunk_size - overlap) <= chunk_size * (1+chunk_size_tol)
				): # if previous+this chunk in same page/layout is small, merge them
					this_chunk_size = prev_chunk_size + this_chunk_size - overlap
					words_lst[-1].extend(chunk_words[overlap:])
					boxes_lst[-1].extend(chunk_boxes[overlap:])
					stats["chunk_size_dist"][prev_chunk_size] -= 1
					try:
						stats_examples["chunk_size_dist"][prev_chunk_size].remove(f"{question_id[b]}_p{p}")
					except ValueError:
						pass
					stats["chunk_size_dist"][this_chunk_size] += 1
					self.stat_add_example(stats_examples["chunk_size_dist"], this_chunk_size, f"{question_id[b]}_p{p}")
				else:
					p_lst.append(_p)
					words_lst.append(chunk_words)
					boxes_lst.append(chunk_boxes)
					stats["chunk_size_dist"][len(chunk_words)] += 1
					self.stat_add_example(stats_examples["chunk_size_dist"], len(chunk_words), f"{question_id[b]}_p{p}")
					n_chunks += 1
				prev_chunk_size = this_chunk_size
			return n_chunks

		for b, (batch_words, batch_boxes) in enumerate(zip(words, boxes)): # (n_pages, n_words), (n_pages, n_words, 4)
			if layout_boxes:
				batch_layout_boxes = layout_boxes[b]
				batch_layout_labels = layout_labels[b]
			else:
				batch_layout_boxes = None
				batch_layout_labels = None
			batch_text_chunks = [] # (n_chunks,)
			batch_box_chunks = [] # (n_chunks, 4)
			batch_page_indices = [] # (n_chunks,)
			batch_words_text_chunks = [] # (n_chunks, n_words)
			batch_words_box_chunks = [] # (n_chunks, n_words, 4)
			batch_layout_labels_chunks = []
			batch_n_chunks = 0
			for p, (page_words, page_boxes) in enumerate(zip(batch_words, batch_boxes)): # (n_words,), (n_words, 4)
				if not isinstance(page_words, list):
					page_boxes = page_boxes.tolist()
				if self.page_retrieval == "oracle":
					# If oracle, take the whole page as a chunk
					batch_page_indices.append(p)
					batch_words_text_chunks.append(page_words)
					batch_words_box_chunks.append(page_boxes)
					batch_layout_labels_chunks.append(10) # 10 = "text"
					stats["chunk_size_dist"][len(page_words)] += 1
					stats["n_chunks_per_page_dist"][1] += 1
					batch_n_chunks += 1
					self.stat_add_example(stats_examples["chunk_size_dist"], len(page_words), f"{question_id[b]}_p{p}")
					self.stat_add_example(stats_examples["n_chunks_per_page_dist"], 1, f"{question_id[b]}_p{p}")
				elif not (batch_layout_boxes and batch_layout_boxes[p]): # (AnyConfOracle is included here)
					# Else, if no layout, make chunks inside the page
					page_n_chunks = make_chunks(
						page_words, page_boxes, p,
						batch_words_text_chunks, batch_words_box_chunks, batch_page_indices
					)
					batch_layout_labels_chunks.extend([10] * page_n_chunks) # 10 = "text"
					batch_n_chunks += page_n_chunks
					stats["n_chunks_per_page_dist"][page_n_chunks] += 1
					self.stat_add_example(stats_examples["n_chunks_per_page_dist"], page_n_chunks, f"{question_id[b]}_p{p}")
				else:
					# Else, if layout, make chunks inside the layout boxes
					page_layout_boxes = batch_layout_boxes[p]
					page_layout_labels = batch_layout_labels[p]
					layout_words_text = [] # (n_chunks, n_words)
					layout_words_boxes = [] # (n_chunks, n_words, 4)
					layout_indices = [] # (n_chunks,)
					page_n_chunks = 0
					for lb, (layout_box, layout_label) in enumerate(zip(page_layout_boxes, page_layout_labels)):
						# Find words inside the layout box
						words_inside = []
						boxes_inside = []
						for word, box in zip(page_words, page_boxes):
							contain_ratio = self.layout_model._containment_ratio(box, layout_box)
							if contain_ratio > 0.5:
								words_inside.append(word)
								boxes_inside.append(box)
						# Split the words inside the layout box into chunks
						layout_n_chunks = make_chunks(
							words_inside, boxes_inside, lb,
							layout_words_text, layout_words_boxes, layout_indices
						)
						page_n_chunks += layout_n_chunks
						batch_layout_labels_chunks.extend([layout_label] * layout_n_chunks)
						stats["n_chunks_per_layout_dist"][layout_n_chunks] += 1
						self.stat_add_example(stats_examples["n_chunks_per_layout_dist"], layout_n_chunks, f"{question_id[b]}_p{p}")
					batch_page_indices.extend([p] * len(layout_words_text))
					batch_words_text_chunks.extend(layout_words_text)
					batch_words_box_chunks.extend(layout_words_boxes)
					batch_n_chunks += page_n_chunks
					stats["n_chunks_per_page_dist"][page_n_chunks] += 1
					self.stat_add_example(stats_examples["n_chunks_per_page_dist"], page_n_chunks, f"{question_id[b]}_p{p}")
			
			# Join words and boxes for each chunk
			for chunk_words, chunk_boxes in zip(batch_words_text_chunks, batch_words_box_chunks):
				batch_text_chunks.append(" ".join(chunk_words))
				if self.page_retrieval == "oracle":
					batch_box_chunks.append([0, 0, 1, 1])
				else:
					# Find box of the chunk
					min_x = min([box[0] for box in chunk_boxes])
					min_y = min([box[1] for box in chunk_boxes])
					max_x = max([box[2] for box in chunk_boxes])
					max_y = max([box[3] for box in chunk_boxes])
					batch_box_chunks.append([min_x, min_y, max_x, max_y])

			if not return_words:
				batch_words_text_chunks = []
				batch_words_box_chunks = []
			
			text_chunks.append(batch_text_chunks)
			boxes_chunks.append(batch_box_chunks)
			page_indices.append(batch_page_indices)
			words_text_chunks.append(batch_words_text_chunks)
			words_boxes_chunks.append(batch_words_box_chunks)
			layout_labels_chunks.append(batch_layout_labels_chunks)
			stats["n_chunks_per_doc_dist"][batch_n_chunks] += 1
			self.stat_add_example(stats_examples["n_chunks_per_doc_dist"], batch_n_chunks, f"{question_id[b]}")
		return text_chunks, boxes_chunks, page_indices, words_text_chunks, words_boxes_chunks, layout_labels_chunks, stats, stats_examples

	@no_grad()
	def retrieve(
			self,
			batch: dict,
			k: int = 5,
			chunk_size: int = 30,
			chunk_size_tol: float = 0.15,
			overlap: int = 0,
			return_words: bool = False,
			include_surroundings: int = 0
	) -> tuple:
		"""
		Retrieve top k chunks and corresponding image patches
			:param batch: input batch with "question", "words", "boxes" and "images"
			:param k: number of chunks to retrieve
			:param chunk_size: size of the chunk in words
			:param overlap: overlap of words between chunks
			:param return_words: if True, return also the ocr and boxes for individual words
			:param include_surroundings: number of words to include before and after each chunk
			:return: top k chunks, top k boxes, top k image patches, top k page indices,
					top k words text chunks, top k words box chunks, top k layout labels,
					similarities, layout_info, stats
		"""
		# Should take [question, pages ocr, pages boxes, pages images]
		# and return [question, chunks ocr, chunks boxes, chunks image patches]
		questions = batch["questions"] # (bs, )
		words = batch["words"] # (bs, n_pages, n_words)
		boxes = batch["boxes"] # (bs, n_pages, n_words, 4)
		images = batch["images"] # (bs, n_pages) PIL images
		bs = len(questions)

		stats = {}
		stats_examples = {}

		# Get layout boxes and labels
		start_time = time()
		if self.layout_model is not None:
			flatten_images = []  # (bs*n_pages,)
			for b in range(bs):
				page_images = images[b]
				flatten_images.extend(page_images)
			
			# Divide into batches of layout_bs
			new_batches = []  # (bs_l, n_pages_l)
			for i in range(0, len(flatten_images), self.layout_bs):
				batch_images = flatten_images[i:i + self.layout_bs]
				new_batches.append(batch_images)
			
			# Process batches and flatten again
			flatten_layout_info = []  # (bs*n_pages,)
			flatten_layout_segments = []  # (bs*n_pages, h, w)
			flatten_layout_info_raw = []  # (bs*n_pages,)
			for batch_images in new_batches:
				batch_layout_boxes, steps = self.layout_model(batch_images, return_steps=True)
				flatten_layout_info.extend(batch_layout_boxes)
				flatten_layout_segments.extend(steps["segmentation"])
				flatten_layout_info_raw.extend(steps["layout_info_raw"])
			
			# Reshape flatten_layout_boxes back to (bs, n_pages, n_boxes, 4)
			layout_info = [] # (bs, n_pages)
			layout_segments = [] # (bs, n_pages, h, w)
			layout_info_raw = [] # (bs, n_pages)
			# layout_info[b][p]: {"boxes": (n_boxes, 4), "labels": (n_boxes,)}
			index = 0
			for b in range(bs):
				page_layouts = []
				page_segments = []
				page_info_raw = []
				for p in range(len(images[b])):
					page_layouts.append(flatten_layout_info[index])
					page_segments.append(flatten_layout_segments[index])
					page_info_raw.append(flatten_layout_info_raw[index])
					index += 1
				layout_info.append(page_layouts)
				layout_segments.append(page_segments)
				layout_info_raw.append(page_info_raw)
		else:
			layout_info = [[]]
			layout_segments = [[]]
			layout_info_raw = [[]]
		stats["layout_time"] = time() - start_time

		# Get chunks
		text_chunks, box_chunks, page_indices, words_text_chunks, words_box_chunks, layout_labels_chunks, chunk_stats, chunk_stats_examples = \
			self.get_chunks(words, boxes, layout_info, chunk_size, chunk_size_tol, overlap, return_words, question_id=batch["question_id"])

		# Compute some statistics
		if layout_info != [[]]:
			stats["n_layouts_per_page_dist"] = Counter()
			stats["layouts_size_w_dist"] = Counter()
			stats["layouts_size_h_dist"] = Counter()
			stats["layout_labels_dist"] = {layout_map[label]: 0 for label in layout_map}
			stats_examples["n_layouts_per_page_dist"] = {}
			stats_examples["layouts_size_w_dist"] = {}
			stats_examples["layouts_size_h_dist"] = {}
			for b in range(bs):
				for p in range(len(layout_info[b])):
					# n_layouts_per_page_dist
					n_layouts = len(layout_info[b][p]["boxes"])
					stats["n_layouts_per_page_dist"][n_layouts] += 1
					self.stat_add_example(stats_examples["n_layouts_per_page_dist"], n_layouts, f"{batch['question_id'][b]}_p{p}")
					# layouts_size_w_dist, layouts_size_h_dist
					for box in layout_info[b][p]["boxes"]:
						w = box[2] - box[0]
						h = box[3] - box[1]
						stats["layouts_size_w_dist"][w] += 1
						stats["layouts_size_h_dist"][h] += 1
						self.stat_add_example(stats_examples["layouts_size_w_dist"], w, f"{batch['question_id'][b]}_p{p}")
						self.stat_add_example(stats_examples["layouts_size_h_dist"], h, f"{batch['question_id'][b]}_p{p}")
					# layout_labels_dist
					for label in layout_info[b][p]["labels"]:
						stats["layout_labels_dist"][layout_map[label]] += 1
		stats.update(chunk_stats)
		stats_examples.update(chunk_stats_examples)

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
		top_k_layout_labels = []  # (bs, k)

		for b in range(bs):
			k_min = min(k, len(similarities[b]))
			top_k = torch.topk(similarities[b], k=k_min, dim=-1).indices

			# these do not contain surrounding words, just the retrieved chunks
			top_k_text.append([text_chunks[b][i] for i in top_k])
			top_k_boxes.append([box_chunks[b][i] for i in top_k])
			top_k_page_indices.append([page_indices[b][i] for i in top_k])
			top_k_layout_labels.append([layout_labels_chunks[b][i] for i in top_k])

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
					words_text = [page_words_text[page_idx][idx] for idx in new_word_indices] # (n_words,)
					words_boxes = [page_words_boxes[page_idx][idx] for idx in new_word_indices]

					# Append to batch results
					batch_top_k_words_text.append(words_text) # (k, n_words)
					batch_top_k_words_boxes.append(words_boxes)

				# Append batch data to the final results
				top_k_words_text.append(batch_top_k_words_text) # (bs, k, n_words)
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

		# Organize output
		if self.page_retrieval == "oracle":
			top_k_page_indices = [[batch["answer_page_idx"][b]] for b in range(bs)]
		elif self.page_retrieval == "anyconforacle":
			top_k_page_indices = [[batch["answer_page_idx"][b]] * len(top_k_text[b]) for b in range(bs)]

		steps = {
			"text_chunks": text_chunks, # (bs, n_chunks)
			"similarities": similarities, # (bs, n_chunks)
			"layout_info": layout_info, # (bs, n_pages)
			"layout_segments": layout_segments, # (bs, n_pages, h, w)
			"layout_info_raw": layout_info_raw # (bs, n_pages)
		}

		stats["layout_labels_topk_dist"] = {label: 0 for label in layout_map.values()}
		for b in range(bs):
			for c in range(len(top_k_layout_labels[b])):
				label = top_k_layout_labels[b][c]
				stats["layout_labels_topk_dist"][layout_map[label]] += 1

		stats = {"stats": stats, "stats_examples": stats_examples}

		return (
			top_k_text, # (bs, k)
			top_k_boxes, # (bs, k, 4)
			top_k_patches, # (bs, k, h, w, 3)
			top_k_page_indices, # (bs, k)
			top_k_words_text, # (bs, k, n_words)
			top_k_words_boxes, # (bs, k, n_words, 4)
			top_k_layout_labels, # (bs, k)
			steps, # dict
			stats # dict
		)

	def forward(
			self,
			batch: dict,
			return_pred_answer: bool = True,
			return_retrieval: bool = False,
			chunk_num: int = 5,
			chunk_size: int = 30,
			chunk_size_tol: float = 0.15,
			overlap: int = 0,
			include_surroundings: int = 0
	) -> tuple:
		# Retrieve top k chunks and corresponding image patches
		start_time = time()
		(
			top_k_text,
			top_k_boxes,
			top_k_patches,
			top_k_page_indices,
			top_k_words_text,
			top_k_words_boxes,
			top_k_layout_labels,
			steps,
			stats
		) = self.retrieve(batch, chunk_num, chunk_size, chunk_size_tol, overlap, True, include_surroundings)
		retrieval_time = time() - start_time
		bs = len(top_k_text)
		similarities = steps["similarities"]
		# Generate
		start_time = time()
		new_batch = {}
		if self.page_retrieval in ["oracle", "concat"]:
			# Concatenate all the top k chunks (in case of oracle, just 1 chunk, the whole page)
			new_batch["questions"] = batch["questions"].copy() # (bs,)
			new_batch["words"] = [flatten(b, self.add_sep_token) for b in top_k_words_text]  # (bs, k * n_words)
			new_batch["boxes"] = [flatten(b, self.add_sep_token) for b in top_k_words_boxes]  # (bs, k * n_words, 4)
			new_batch["images"] = [concatenate_patches(b, mode="grid") for b in top_k_patches]  # (bs, h, w, 3)
			new_batch["answers"] = batch["answers"].copy()  # (bs, n_answers)
			result = self.generator(new_batch, return_pred_answer=return_pred_answer)  # (4, bs)
		elif self.page_retrieval in ("maxconf", "anyconf", "maxconfpage", "anyconfpage", "anyconforacle"):
			# Generate for each top k chunk
			results = []  # (bs, 4, k)
			max_confidence_indices = []
			for b in range(bs):  # iterate over batch

				words, boxes, patches = [], [], []
				if self.page_retrieval in ["maxconf", "anyconf", "anyconforacle"]:
					# Prepare the data for each chunk
					for i in range(len(top_k_words_text[b])):
						# Filter out empty chunks
						if top_k_words_text[b][i] != []:
							words.append(top_k_words_text[b][i])
							boxes.append(top_k_words_boxes[b][i])
							patches.append(top_k_patches[b][i])
				elif self.page_retrieval in ["maxconfpage", "anyconfpage"]:
					# Prepare the data for the page corresponding to each chunk
					for i in range(len(top_k_page_indices[b])):
						page_idx = top_k_page_indices[b][i]
						words.append(batch["words"][b][page_idx])
						boxes.append(batch["boxes"][b][page_idx])
						patches.append(batch["images"][b][page_idx])

				if len(words) == 0:
					# If there are no words, append None and continue
					results.append(None)
					max_confidence_indices.append(None)
					continue

				# Prepare new batch
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

			# Format results from (bs, 4, k) to (4, bs) for Maxconf and (4, bs, k) for Anyconf
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
						if i == 0: # not interested in loss, logits, etc.
							final_results[i].append(None)
						else:
							if self.page_retrieval in ("maxconf", "maxconfpage"):
								# Take the answer with highest confidence
								final_results[i].append(result[i][conf_idx])
							elif self.page_retrieval in ("anyconf", "anyconfpage", "anyconforacle"):
								# Take all answers
								final_results[i].append(result[i])
					else:
						final_results[i].append(None)
			result = tuple(final_results)
		elif self.page_retrieval in ["majorpage", "weightmajorpage"]:
			# Prepare the data for the page corresponding to the majority voted chunks
			# e.g. if chunk pages are [0, 0, 1, 2, 2] and weights are [0.5, 0.5, 0.2, 0.3, 0.3]
			# then the majority voted page is 0
			if self.page_retrieval == "majorpage":
				weights = [np.ones(len(similarities[b])) for b in range(bs)]
			else:
				weights = [similarities[b].cpu().numpy() for b in range(bs)]
			weights = [w / sum(w) for w in weights]  # Normalize weights

			# Prepare new batch
			major_page_indices = [] # (bs,)
			for b in range(bs):
				page_indices_b = top_k_page_indices[b]
				weights_b = weights[b]
				unique_pages = list(set(page_indices_b))
				page_weights = {page: 0 for page in unique_pages}
				for page, weight in zip(page_indices_b, weights_b):
					page_weights[page] += weight
				if len(page_weights) == 0:
					major_page_indices.append(0)
					continue
				major_page_indices.append(max(page_weights, key=page_weights.get))
			new_batch["questions"] = batch["questions"].copy()  # (bs,)
			new_batch["words"] = [batch["words"][b][page_idx] for b, page_idx in enumerate(major_page_indices)]  # (bs, n_words)
			new_batch["boxes"] = [batch["boxes"][b][page_idx] for b, page_idx in enumerate(major_page_indices)]  # (bs, n_words, 4)
			new_batch["images"] = [batch["images"][b][page_idx] for b, page_idx in enumerate(major_page_indices)]  # (bs, h, w, 3)
			new_batch["answers"] = batch["answers"].copy()  # (bs, n_answers)
			# This treats bs as batch size
			result = self.generator(new_batch, return_pred_answer=return_pred_answer)  # (4, bs)			

		generation_time = time() - start_time

		if return_retrieval:
			# For visualization
			retrieval = {
				"text": top_k_text,
				"boxes": top_k_boxes,
				"patches": top_k_patches,
				"page_indices":
					top_k_page_indices if self.page_retrieval not in ["majorpage", "weightmajorpage"]
					else major_page_indices,
				"words_text": top_k_words_text,
				"words_boxes": top_k_words_boxes,
				"top_k_layout_labels": top_k_layout_labels,
				"steps": steps,
				"stats": stats["stats"],
				"stats_examples": stats["stats_examples"],
				"retrieval_time": retrieval_time,
				"generation_time": generation_time
			}
			if self.page_retrieval in ["oracle", "concat"]:
				retrieval.update({
					"input_words": new_batch["words"],
					"input_boxes": new_batch["boxes"],
					"input_patches": new_batch["images"]
				})
			elif self.page_retrieval in ["maxconf", "anyconf", "maxconfpage", "anyconfpage", "anyconforacle"]:
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
			**kwargs
	):
		self.eval()
		return self.forward(
			batch,
			**kwargs
		)
