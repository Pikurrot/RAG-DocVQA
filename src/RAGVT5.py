import torch
from src.VT5 import VT5ForConditionalGeneration
from src ._modules import CustomT5Config, Chunker, Embedder, Retriever, LayoutModel
from src.utils import flatten, concatenate_patches
from typing import Tuple, Any
import numpy as np
from time import time
from collections import Counter

class RAGVT5:
	def __init__(self, config: dict):
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

		# Load components
		#	Layout model
		if self.layout_model_weights is not None:
			self.layout_model = LayoutModel(config)
		else:
			self.layout_model = None

		# 	Chunker
		self.chunker = Chunker(config)

		# 	Embedder
		self.embedder = Embedder(config, language_model=self.generator.language_backbone if self.embed_model == "VT5" else None)

		# 	Retriever
		self.retriever = Retriever(config)

		# 	Generator
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

	def to(self, device: Any):
		self.device = device
		self.generator.to(device)
		self.embedder.to(device)
		self.layout_model.to(device)

	def eval(self):
		self.generator.eval()

	def train(self):
		self.generator.train()

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
		layout_labels_chunks = [] # (bs, n_chunks)
		page_indices = [] # (bs, n_chunks)
		words_text_chunks = [] # (bs, n_chunks, n_words)
		words_boxes_chunks = [] # (bs, n_chunks, n_words, 4)
		words_layout_labels_pages = [] # (bs, n_pages, n_words)
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
			batch_layout_labels_chunks = [] # (n_chunks,)
			batch_page_indices = [] # (n_chunks,)
			batch_words_text_chunks = [] # (n_chunks, n_words)
			batch_words_box_chunks = [] # (n_chunks, n_words, 4)
			batch_words_layout_labels_pages = [] # (n_pages, n_words)
			batch_n_chunks = 0
			for p, (page_words, page_boxes) in enumerate(zip(batch_words, batch_boxes)): # (n_words,), (n_words, 4)
				if not isinstance(page_words, list):
					page_boxes = page_boxes.tolist()
				if len(page_boxes) > 0 and not isinstance(page_boxes[0], list):
					page_boxes = [pbox.tolist() for pbox in page_boxes]
				if not (batch_layout_boxes and batch_layout_boxes[p]):
					# If no layout, make chunks inside the page
					page_n_chunks = make_chunks(
						page_words, page_boxes, p,
						batch_words_text_chunks, batch_words_box_chunks, batch_page_indices
					)
					batch_layout_labels_chunks.extend([10] * page_n_chunks) # 10 = "text"
					batch_words_layout_labels_pages.append([10] * len(page_words))
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
					page_words_layout_labels = [10] * len(page_words) # (n_words,)
					for lb, (layout_box, layout_label) in enumerate(zip(page_layout_boxes, page_layout_labels)):
						# Find words inside the layout box
						words_inside = []
						boxes_inside = []
						for i, (word, box) in enumerate(zip(page_words, page_boxes)):
							contain_ratio = self.layout_model.containment_ratio(box, layout_box)
							if contain_ratio > 0.5:
								words_inside.append(word)
								boxes_inside.append(box)
								page_words_layout_labels[i] = layout_label
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
					batch_words_layout_labels_pages.append(page_words_layout_labels)
					batch_n_chunks += page_n_chunks
					stats["n_chunks_per_page_dist"][page_n_chunks] += 1
					self.stat_add_example(stats_examples["n_chunks_per_page_dist"], page_n_chunks, f"{question_id[b]}_p{p}")
			
			# If oracle, take the whole page as a chunk (actually join the chunks of the page)
			if self.page_retrieval == "oracle":
				batch_words_text_chunks = [flatten(batch_words_text_chunks)]
				batch_words_box_chunks = [flatten(batch_words_box_chunks)]
				batch_layout_labels_chunks = [10]
				batch_page_indices = [0]
				batch_n_chunks = 1
			
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
			layout_labels_chunks.append(batch_layout_labels_chunks)
			page_indices.append(batch_page_indices)
			words_text_chunks.append(batch_words_text_chunks)
			words_boxes_chunks.append(batch_words_box_chunks)
			words_layout_labels_pages.append(batch_words_layout_labels_pages)
			stats["n_chunks_per_doc_dist"][batch_n_chunks] += 1
			self.stat_add_example(stats_examples["n_chunks_per_doc_dist"], batch_n_chunks, f"{question_id[b]}")
		return text_chunks, boxes_chunks, layout_labels_chunks, page_indices, words_text_chunks, words_boxes_chunks, words_layout_labels_pages, stats, stats_examples

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

		stats = {}
		stats_examples = {}

		# Get layout boxes and labels
		start_time = time()
		if self.layout_model is not None:
			layout_info, layout_steps = self.layout_model.batch_forward(images, return_steps=True, question_id=batch["question_id"])
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
		text_chunks, _ = Chunker.compact_chunks(words_text_chunks, words_boxes_chunks)

		# Get text and question embeddings
		text_embeddings = self.embedder.embed_multi(text_chunks) # (bs, n_chunks, hidden_size)
		question_embeddings = self.embedder.embed(questions) # (bs, hidden_size)

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
			similarities,
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
		if self.page_retrieval in ["oracle", "concat"]:
			# Concatenate all the top k chunks (in case of oracle, just 1 chunk, the whole page)
			new_batch["questions"] = batch["questions"].copy() # (bs,)
			new_batch["words"] = [flatten(b, self.add_sep_token) for b in top_k_words_text]  # (bs, k * n_words)
			new_batch["boxes"] = [flatten(b, self.add_sep_token) for b in top_k_words_boxes]  # (bs, k * n_words, 4)
			new_batch["layout_labels"] = [flatten(b, self.add_sep_token) for b in top_k_words_layout_labels]  # (bs, k * n_words)
			new_batch["images"] = [concatenate_patches(b, mode="grid") for b in top_k_patches]  # (bs, h, w, 3)
			new_batch["answers"] = batch["answers"].copy()  # (bs, n_answers)
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

	@torch.inference_mode
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
