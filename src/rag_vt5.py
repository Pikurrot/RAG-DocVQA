import torch
from src.VT5 import VT5ForConditionalGeneration
from src ._modules import CustomT5Config
from src._model_utils import torch_no_grad, mean_pooling
from typing import Tuple
from PIL import Image
import numpy as np

class RAGVT5(torch.nn.Module):
	def __init__(
			self,
			config: dict,
	):
		super(RAGVT5, self).__init__()

		# Load config
		self.save_dir = config.get("save_dir", "save/")
		self.batch_size = config.get("batch_size", 16)
		self.model_path = config.get("model_weights", "rubentito/vt5-base-spdocvqa")
		self.page_retrieval = config["page_retrieval"].lower() if "page_retrieval" in config else None
		self.max_source_length = config.get("max_source_length", 512)
		self.device = config.get("device", "cuda")

		t5_config = CustomT5Config.from_pretrained(self.model_path, ignore_mismatched_sizes=True)
		t5_config.visual_module_config = config.get("visual_module", {})

		# Load models
		self.generator = VT5ForConditionalGeneration.from_pretrained(
			self.model_path, config=t5_config, ignore_mismatched_sizes=True
		)
		self.generator.load_config(config)

	def to(self, device):
		self.device = device
		self.generator.to(device)

	def eval(self):
		self.generator.eval()

	def train(self):
		self.generator.train()

	def get_chunks(
			self,
			words: list, # (bs, n_pages, n_words)
			boxes: list, # (bs, n_pages, n_words, 4)
			chunk_size: int = 30,
			overlap: int = 0
	) -> Tuple[list, list ,list]: # (bs, n_chunks), (bs, n_chunks, 4), (bs, n_chunks)
		"""
		Converts words and boxes to chunks
			:param words: list of words
			:param boxes: list of boxes
			:param chunk_size: size of the chunk in words
			:param overlap: overlap of words between chunks
			:return: text chunks, box chunks, corresponding page indices
		"""
		assert chunk_size > 1, "chunk_size should be a non-negative non-zero integer."
		assert overlap >= 0, "overlap should be a non-negative integer."
		assert overlap < chunk_size, "overlap should be less than chunk_size."

		text_chunks = []
		box_chunks = []
		page_indices = []
		for b, (batch_words, batch_boxes) in enumerate(zip(words, boxes)):
			batch_text_chunks = []
			batch_box_chunks = []
			batch_page_indices = []
			for p, (page_words, page_boxes) in enumerate(zip(batch_words, batch_boxes)):
				# Split the page words and boxes into chunks
				for i in range(0, len(page_words), chunk_size - overlap):
					chunk_text = " ".join(page_words[i:i + chunk_size])
					chunk_boxes = page_boxes[i:i + chunk_size]
					batch_text_chunks.append(chunk_text)
					# Find box of the chunk
					min_x = min([box[0] for box in chunk_boxes])
					min_y = min([box[1] for box in chunk_boxes])
					max_x = max([box[2] for box in chunk_boxes])
					max_y = max([box[3] for box in chunk_boxes])
					batch_box_chunks.append([min_x, min_y, max_x, max_y])
					batch_page_indices.append(p)
			text_chunks.append(batch_text_chunks)
			box_chunks.append(batch_box_chunks)
			page_indices.append(batch_page_indices)
		return text_chunks, box_chunks, page_indices

	@torch_no_grad
	def retrieve(
			self,
			batch: dict,
			k: int = 5,
			chunk_size: int = 30,
			overlap: int = 0
	) -> Tuple[list, list, list, list]: # (bs, k), (bs, k, 4), (bs, k), (bs, k, h, w, 3)
		"""
		Retrieve top k chunks and corresponding image patches
			:param batch: input batch with "question", "words", "boxes" and "images"
			:param k: number of chunks to retrieve
			:param chunk_size: size of the chunk in words
			:param overlap: overlap of words between chunks
			:return: top k chunks, top k boxes, top k image patches, top k page indices
		"""
		# Should take [question, pages ocr, pages boxes, pages images]
		# and return [question, chunks ocr, chunks boxes, chunks image patches]
		questions = batch["questions"] # (bs, )
		words = batch["words"] # (bs, n_pages, n_words)
		boxes = batch["boxes"] # (bs, n_pages, n_words, 4)
		images = batch["images"] # (bs, n_pages) PIL images
		bs = len(questions)

		# Get chunks
		text_chunks, box_chunks, page_indices = self.get_chunks(words, boxes, chunk_size, overlap)

		# Get text embeddings
		text_embeddings = [] # (bs, n_chunks, hidden_size)
		for text_batch in text_chunks:
			input_ids, attention_mask = self.generator.tokenizer(
				text_batch,
				return_tensors='pt',
				padding=True,
				truncation=True
			).values()
			input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
			text_tokens_embeddings = self.generator.language_backbone.shared(input_ids)
			text_embeddings.append(mean_pooling(text_tokens_embeddings, attention_mask))
		print(input_ids.shape)
		print(text_embeddings[-1].shape)

		# Get question embeddings
		input_ids, attention_mask = self.generator.tokenizer(
			questions,
			return_tensors='pt',
			padding=True,
			truncation=True
		).values()
		input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
		question_tokens_embeddings = self.generator.language_backbone.shared(input_ids)
		question_embeddings = mean_pooling(question_tokens_embeddings, attention_mask) # (bs, hidden_size)
		print(input_ids.shape)
		print(question_embeddings.shape)

		# Compute similarity
		similarities = [] # (bs, n_chunks)
		norms_quest = torch.norm(question_embeddings, dim=-1)
		for b in range(bs):
			batch_norms_text = torch.norm(text_embeddings[b], dim=-1)
			print(norms_quest.shape, batch_norms_text.shape, text_embeddings[b].shape, question_embeddings[b].shape)
			similarity = torch.matmul(text_embeddings[b], question_embeddings[b]) / (batch_norms_text * norms_quest[b])
			similarities.append(similarity)

		# Get top k chunks and boxes
		top_k_chunks = [] # (bs, k)
		top_k_boxes = [] # (bs, k, 4)
		top_k_page_indices = [] # (bs, k)
		for b in range(bs):
			k_min = min(k, len(similarities[b]))
			top_k = torch.topk(similarities[b], k=k_min, dim=-1).indices
			top_k_chunks.append([text_chunks[b][i] for i in top_k])
			top_k_boxes.append([box_chunks[b][i] for i in top_k])
			top_k_page_indices.append([page_indices[b][i] for i in top_k])

		# Get image patches
		top_k_patches = [] # (bs, k, h, w, 3)
		for b in range(bs):
			batch_patches = []
			for i, page_idx in enumerate(top_k_page_indices[b]):
				page: Image.Image = images[b][page_idx] # (H, W, 3)
				box: np.ndarray = top_k_boxes[b][i] # (4,)
				# transform to absolute coordinates
				box[0] = int(box[0] * page.width)
				box[1] = int(box[1] * page.height)
				box[2] = int(box[2] * page.width)
				box[3] = int(box[3] * page.height)
				patch = page.crop(box) # (h, w, 3)
				batch_patches.append(patch)
			top_k_patches.append(batch_patches)

		return top_k_chunks, top_k_boxes, top_k_patches, top_k_page_indices
			

	def forward(self, batch: dict, return_pred_answer: bool=True):
		# Retrieve top k chunks and corresponding image patches
		top_k_chunks, top_k_boxes, top_k_patches, _ = self.retrieve(batch, k=5)
		new_batch = batch.copy()
		new_batch["words"] = top_k_chunks
		new_batch["boxes"] = top_k_boxes
		new_batch["images"] = top_k_patches
		# Generate
		return self.generator(new_batch, return_pred_answer=return_pred_answer)
		