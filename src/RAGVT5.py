import torch
import numpy as np
from src._modules import (
	CustomT5Config,
	Chunker,
	BiEncoder,
	Retriever,
	Reranker,
	LayoutModel,
	get_layout_model_map
)
from src.utils import flatten, concatenate_patches
from typing import Any
from time import time
from contextlib import nullcontext
from transformers import Qwen2_5_VLConfig
from src.VT5 import VT5ForConditionalGeneration
from src.QwenVLInstruct import QwenVLForConditionalGeneration

class RAGVT5(torch.nn.Module):
	def __init__(self, config: dict):
		super(RAGVT5, self).__init__()
		# Load config
		self.model_path = config.get("model_weights", "rubentito/vt5-base-spdocvqa")
		self.embed_path = config.get("embed_weights", None)
		self.layout_bs = config.get("layout_batch_size", 1)
		self.use_precomputed_layouts = config.get("use_precomputed_layouts", False)
		self.use_layout_labels = config.get("use_layout_labels", "Default")
		self.layout_map = get_layout_model_map(config)
		self.page_retrieval = config.get("page_retrieval")
		if self.use_precomputed_layouts or self.page_retrieval == "oracle":
			self.layout_model_weights = None
		else:
			self.layout_model_weights = config.get("layout_model_weights", None)
		print(f"Loading model from {self.model_path}")
		if self.embed_path:
			print(f"Loading embedding model from {self.embed_path}")
		else:
			print("Using the same model for embeddings")
		if self.layout_model_weights:
			print(f"Loading layout model from {self.layout_model_weights}")
		elif self.use_precomputed_layouts:
			print("Using precomputed layouts")
		else:
			print("Not using layout information")
		self.max_source_length = config.get("max_source_length", 512)
		self.device = config.get("device", "cuda")
		self.embed_weights = config.get("embed_weights", "VT5")
		self.reranker_weights = config.get("reranker_weights", None)
		self.add_sep_token = config.get("add_sep_token", False)
		self.cache_dir = config.get("cache_dir", None)
		print(f"Using {self.cache_dir} as cache folder")
		self.train_layout = config.get("train_layout", False)
		self.train_embedder = config.get("train_embed", False)
		self.train_generator = (
			config.get("train_language_backbone", False) or
			config.get("train_spatial_embedding", False) or
			config.get("train_visual_embedding", False) or
			config.get("train_layout_embedding", False)
		)
		self.train_mode = False

		# Load components
		if self.layout_model_weights and self.page_retrieval != "oracle":
			self.layout_model = LayoutModel(config)
		else:
			self.layout_model = None
		self.chunker = Chunker(config)
		self.embedder = BiEncoder(config, language_model=self.generator.language_backbone if self.embed_weights == "VT5" else None)
		if self.reranker_weights:
			self.reranker = Reranker(config)
		else:
			self.reranker = None
		self.retriever = Retriever(config)
		
		if "qwen" in self.model_path.lower():
			qwen_config = Qwen2_5_VLConfig.from_pretrained(self.model_path, cache_dir=self.cache_dir)
			qwen_config.update(config)
			self.generator = QwenVLForConditionalGeneration.from_pretrained(
				self.model_path,
				config=qwen_config,
				torch_dtype=torch.bfloat16,
				attn_implementation="flash_attention_2",
   				device_map="auto",
				cache_dir=self.cache_dir
			)
		else:
			t5_config = CustomT5Config.from_pretrained(self.model_path, ignore_mismatched_sizes=True, cache_dir=self.cache_dir)
			t5_config.visual_module_config = config.get("visual_module", {})
			t5_config.layout_loss_weight = config.get("layout_loss_weight", 1.0)
			t5_config.update(config)
			self.generator = VT5ForConditionalGeneration.from_pretrained(
				self.model_path, config=t5_config, ignore_mismatched_sizes=True, cache_dir=self.cache_dir
			)

		if self.add_sep_token:
			# Add the chunk separator token to the tokenizer if not already present
			token_id = self.generator.tokenizer.encode("<sep>", add_special_tokens=False)
			if not(len(token_id) == 1 and token_id[0] != self.generator.tokenizer.unk_token_id):
				print("Adding <sep> token to the tokenizer. This model should now be fine-tuned.")
				self.generator.tokenizer.add_tokens(["<sep>"])
				self.generator.language_backbone.resize_token_embeddings(len(self.generator.tokenizer))

	def to(self, device: Any):
		self.device = device
		self.generator.to(device)
		self.embedder.to(device)
		if self.layout_model is not None:
			self.layout_model.to(device)

	def eval(self):
		self.generator.eval()
		self.train_mode = False

	def train(self):
		if self.layout_model is not None and self.train_layout:
			self.layout_model.train()
		if self.train_embedder:
			self.embedder.train()
		if self.train_generator:
			self.generator.train()
		self.train_mode = True

	def online_retrieve(
			self,
			batch: dict,
			return_steps: bool = False
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
		if self.layout_model:
			with (nullcontext() if self.train_mode and self.train_layout else torch.no_grad()):
				layout_info, layout_steps = self.layout_model.batch_forward(images, return_steps=True, question_id=batch["question_id"])
		elif self.use_precomputed_layouts:
			layout_info = batch["layouts"] # (bs, n_pages)
			layout_steps = {
				"layout_segments": [[]],
				"layout_info_raw": [[]]
			}
		else:
			layout_info = [[]]
			layout_steps = {
				"layout_segments": [[]],
				"layout_info_raw": [[]]
			}
		stats["layout_time"] = time() - start_time

		# Get chunks
		(
			words_text_chunks, # (bs, n_chunks, n_words)
			words_boxes_chunks, # (bs, n_chunks, n_words, 4)
			layout_labels_chunks, # (bs, n_chunks)
			page_indices, # (bs, n_chunks)
			words_layout_labels_pages, # (bs, n_pages, n_words)
		) =\
			self.chunker.get_chunks(
				words,
				boxes,
				layout_info,
				question_id=batch["question_id"]
		)

		# Prepend layout labels before each chunk if needed
		if self.use_layout_labels == "Text":
			for b in range(bs):
				for i, layout_label in enumerate(layout_labels_chunks[b]):
					text_prepend = self.layout_map[layout_label]+": "
					boxes_prepend = [0, 0, 0, 0]
					words_text_chunks[b][i] = [text_prepend] + words_text_chunks[b][i]
					words_boxes_chunks[b][i] = [boxes_prepend] + words_boxes_chunks[b][i]

		# Compact chunks
		text_chunks, _ = Chunker.compact_chunks(words_text_chunks, words_boxes_chunks)

		# Get text and question embeddings
		with (nullcontext() if self.train_mode and self.train_embedder else torch.no_grad()):
			text_embeddings = self.embedder.batch_forward(text_chunks) # (bs, n_chunks, hidden_size)
			question_embeddings = self.embedder.forward(questions) # (bs, hidden_size)

		# Get top k chunks and boxes
		(
			top_k_text, # (bs, k)
			top_k_boxes, # (bs, k, 4)
			top_k_layout_labels, # (bs, k)
			top_k_words_text, # (bs, k, n_words)
			top_k_words_boxes, # (bs, k, n_words, 4)
			top_k_words_layout_labels, # (bs, k, n_words)
			top_k_patches, # (bs, k, h, w, 3)
			top_k_page_indices, # (bs, k)
			similarities, # (bs, k)
		) =\
			self.retriever.retrieve(
				text_embeddings,
				question_embeddings,
				words_text_chunks,
				words_boxes_chunks,
				layout_labels_chunks,
				images,
				page_indices
		)

		if self.reranker:
			# Rerank
			(
				top_k_text,
				top_k_boxes,
				top_k_layout_labels,
				top_k_words_text,
				top_k_words_boxes,
				top_k_words_layout_labels,
				top_k_patches,
				top_k_page_indices,
				similarities
			) =\
				self.reranker.batch_rerank(
					questions,
					top_k_text,
					top_k_boxes,
					top_k_layout_labels,
					top_k_words_text,
					top_k_words_boxes,
					top_k_words_layout_labels,
					top_k_patches,
					top_k_page_indices,
					similarities
			)

		# Prepare output
		if return_steps:
			steps = {
				"layout_info": layout_info, # (bs, n_pages)
				"text_chunks": text_chunks # (bs, n_chunks)
			}
			steps.update(layout_steps)
		else:
			steps = {}

		if self.layout_model is not None:
			stats.update(self.layout_model.stats)
			stats_examples.update(self.layout_model.stats_examples)
		stats.update(self.chunker.stats)
		stats.update(self.retriever.stats)
		stats_examples.update(self.chunker.stats_examples)
		stats = {"stats": stats, "stats_examples": stats_examples}

		if self.page_retrieval == "oracle":
			top_k_page_indices = [[batch["answer_page_idx"][b]] for b in range(bs)]
		elif self.page_retrieval == "anyconforacle":
			top_k_page_indices = [[batch["answer_page_idx"][b]] * len(top_k_text[b]) for b in range(bs)]

		return (
			top_k_text, # (bs, k)
			top_k_boxes, # (bs, k, 4)
			top_k_layout_labels, # (bs, k)
			top_k_patches, # (bs, k, h, w, 3)
			top_k_page_indices, # (bs, k)
			top_k_words_text, # (bs, k, n_words)
			top_k_words_boxes, # (bs, k, n_words, 4)
			top_k_words_layout_labels, # (bs, k, n_words)
			words_layout_labels_pages, # (bs, n_pages, n_words)
			similarities, # (bs, n_chunks)
			steps, # dict
			stats # dict
		)

	def forward(
			self,
			batch: dict,
			return_pred_answer: bool = True,
			return_retrieval: bool = True,
			return_steps: bool = False
	) -> tuple:
		# Retrieve top k chunks and corresponding image patches
		start_time = time()
		(
			top_k_text,
			top_k_boxes,
			top_k_layout_labels,
			top_k_patches,
			top_k_page_indices,
			top_k_words_text,
			top_k_words_boxes,
			top_k_words_layout_labels,
			words_layout_labels_pages,
			similarities,
			steps,
			stats
		) = self.online_retrieve(batch, return_steps=return_steps)
		retrieval_time = time() - start_time
		bs = len(top_k_text)
		# Generate
		start_time = time()
		new_batch = {}
		if self.use_layout_labels == "Text":
			add_sep_token = "."
		else:
			add_sep_token = "<sep>" if self.add_sep_token else None
		if self.page_retrieval in ["oracle", "concat"]:
			# Concatenate all the top k chunks (in case of oracle, just 1 chunk, the whole page)
			new_batch["questions"] = batch["questions"].copy() # (bs,)
			new_batch["words"] = [flatten(b, add_sep_token) for b in top_k_words_text]  # (bs, k * n_words)
			new_batch["boxes"] = [flatten(b, add_sep_token) for b in top_k_words_boxes]  # (bs, k * n_words, 4)
			new_batch["layout_labels"] = [flatten(b, add_sep_token) for b in top_k_words_layout_labels]  # (bs, k * n_words)
			new_batch["images"] = [concatenate_patches(b, mode="grid") for b in top_k_patches]  # (bs, h, w, 3)
			new_batch["answers"] = batch["answers"].copy()  # (bs, n_answers)
			with (nullcontext() if self.train_mode and self.train_generator else torch.no_grad()):
				result = self.generator(new_batch, return_pred_answer=return_pred_answer)  # (4, bs)
		elif self.page_retrieval in ("maxconf", "anyconf", "maxconfpage", "anyconfpage", "anyconforacle"):
			# Generate for each top k chunk
			results = []  # (bs, 4, k)
			max_confidence_indices = []
			for b in range(bs):  # iterate over batch
				words, boxes, layout_labels, patches = [], [], [], []
				if self.page_retrieval in ["maxconf", "anyconf", "anyconforacle"]:
					# Prepare the data for each chunk
					for i in range(len(top_k_words_text[b])):
						# Filter out empty chunks
						if top_k_words_text[b][i] != []:
							words.append(top_k_words_text[b][i])
							boxes.append(top_k_words_boxes[b][i])
							layout_labels.append(top_k_words_layout_labels[b][i])
							patches.append(top_k_patches[b][i])
				elif self.page_retrieval in ["maxconfpage", "anyconfpage"]:
					# Prepare the data for the page corresponding to each chunk
					for i in range(len(top_k_page_indices[b])):
						page_idx = top_k_page_indices[b][i]
						words.append(batch["words"][b][page_idx])
						boxes.append(batch["boxes"][b][page_idx])
						layout_labels.append(words_layout_labels_pages[b][page_idx])
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
				new_batch["layout_labels"] = layout_labels  # (k, n_words)
				new_batch["images"] = patches  # (k, h, w, 3)
				new_batch["answers"] = [batch["answers"][b]] * len(words)
				# This treats k as batch size
				with (nullcontext() if self.train_mode and self.train_generator else torch.no_grad()):
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
			new_batch["layout_labels"] = [words_layout_labels_pages[b][page_idx] for b, page_idx in enumerate(major_page_indices)]  # (bs, n_words)
			new_batch["images"] = [batch["images"][b][page_idx] for b, page_idx in enumerate(major_page_indices)]  # (bs, h, w, 3)
			new_batch["answers"] = batch["answers"].copy()  # (bs, n_answers)
			# This treats bs as batch size
			with (nullcontext() if self.train_mode and self.train_generator else torch.no_grad()):
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

	@torch.inference_mode()
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
