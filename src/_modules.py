import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config, AutoFeatureExtractor, AutoModel
from sentence_transformers import SentenceTransformer
from torch.nn import LayerNorm as BertLayerNorm
from typing import Optional, Any, Dict, List
from collections import Counter
from PIL import Image
import numpy as np
from src.utils import containment_ratio
from src._model_utils import mean_pooling
from src.LayoutModel import layout_map
from src.VT5 import VT5ForConditionalGeneration

class CustomT5Config(T5Config):
	def __init__(self, max_2d_position_embeddings=1024,  **kwargs):
		super().__init__(**kwargs)
		self.max_2d_position_embeddings = max_2d_position_embeddings
		self.hidden_dropout_prob = 0.1
		self.layer_norm_eps = 1e-12


class SpatialEmbeddings(nn.Module):
	"""
	Spatial embedding by summing x, y, w, h projected by nn.Embedding to hidden size.
	"""

	def __init__(self, config):
		super(SpatialEmbeddings, self).__init__()

		self.x_position_embeddings = nn.Embedding(
			config.max_2d_position_embeddings, config.hidden_size
		)
		self.y_position_embeddings = nn.Embedding(
			config.max_2d_position_embeddings, config.hidden_size
		)

		self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

		self.spatial_emb_matcher = MLP(config.hidden_size, 0, config.hidden_size, 1)

		self.config = config

	def forward(self, bbox):
		left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
		upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
		right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
		lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])

		embeddings = (
				left_position_embeddings
				+ upper_position_embeddings
				+ right_position_embeddings
				+ lower_position_embeddings
		)

		embeddings = self.LayerNorm(embeddings)
		embeddings = self.dropout(embeddings)
		embeddings = self.spatial_emb_matcher(embeddings)
		return embeddings


class MLP(nn.Module):
	""" Very simple multi-layer perceptron (also called FFN)"""

	def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
		super().__init__()
		self.num_layers = num_layers
		h = [hidden_dim] * (num_layers - 1)
		self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

	def forward(self, x):
		for i, layer in enumerate(self.layers):
			x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
		return x


class VisualEmbeddings(nn.Module):

	def __init__(self, config):
		super(VisualEmbeddings, self).__init__()

		self.feature_extractor = AutoFeatureExtractor.from_pretrained(
			config.visual_module_config['model_weights'],
			ignore_mismatched_sizes=True
		)
		self.image_model = AutoModel.from_pretrained(
			config.visual_module_config['model_weights'],
			ignore_mismatched_sizes=True
		)
		self.visual_emb_matcher = MLP(self.image_model.config.hidden_size, 0, self.image_model.config.hidden_size, 1)

		if not config.visual_module_config.get('finetune', False):
			self.freeze()

	def freeze(self):
		for p in self.image_model.parameters():
			p.requires_grad = False

	def get_visual_boxes(self, num_pages=1, scale=1):
		boxes = torch.tensor([[0, 0, 1, 1]] + [[x / 14, y / 14, (x + 1) / 14, (y + 1) / 14] for y in range(0, 14) for x in range(0, 14)], dtype=torch.float32)
		boxes = boxes.unsqueeze(dim=0).expand([num_pages, -1, -1])
		boxes = boxes * scale
		return boxes

	def forward(self, images, page_idx_mask=None):
		inputs = self.feature_extractor(images=images, return_tensors="pt")
		vis_embeddings = self.image_model(inputs.pixel_values.to(self.image_model.device))
		vis_embeddings = vis_embeddings.last_hidden_state  # BS; 14x14+CLS (197); 768 (hidden size)
		vis_embeddings = self.visual_emb_matcher(vis_embeddings)

		if page_idx_mask is not None:
			vis_attention_mask = torch.zeros(vis_embeddings.shape[:2], dtype=torch.long).to(self.image_model.device)
			vis_attention_mask[page_idx_mask] = 1
		else:
			vis_attention_mask = torch.ones(vis_embeddings.shape[:2], dtype=torch.long).to(self.image_model.device)

		return vis_embeddings, vis_attention_mask


class StatComponent:
	def __init__(self, config: dict):
		self.compute_stats = config["compute_stats"]
		self.compute_stats_examples = config["compute_stats_examples"]
		self.n_stats_examples = config["n_stats_examples"]
		self.stats: Dict[str, Any] = {}
		self.stats_examples: Dict[str, Dict[Any, list]] = {}

	def stat_sum(
			self,
			stat: str,
			key: Any,
			value: int=1
	):
		"""
		Add a value to a dictionary.
		"""
		if not self.compute_stats:
			return
		if key not in self.stats[stat]:
			self.stats[stat][key] = 0
		self.stats[stat][key] += value

	def stat_subtract(
			self,
			stat: str,
			key: Any,
			value: int=1
	):
		"""
		Subtract a value from a dictionary.
		"""
		return self.stat_sum(stat, key, -value)

	def stat_add_example(
			self,
			stat: str,
			key: Any,
			example: Any
	):
		"""
		Add an example value to a list in a dictionary.
		"""
		if not self.compute_stats_examples:
			return
		if key not in self.stats_examples[stat]:
			self.stats_examples[stat][key] = []
		if len(self.stats_examples[stat][key]) < self.n_stats_examples:
			self.stats_examples[stat][key].append(example)

	def stat_remove_example(
			self,
			stat: str,
			key: Any,
			example: Any
	):
		"""
		Remove an example value from a list in a dictionary.
		"""
		if not self.compute_stats_examples:
			return
		if key in self.stats_examples[stat]:
			try:
				self.stats_examples[stat][key].remove(example)
			except ValueError:
				pass


class Chunker(StatComponent):
	def __init__(self, config: dict):
		# Load config
		super(Chunker, self).__init__(config)
		self.chunk_size = config["chunk_size"]
		self.chunk_size_tol = config["chunk_size_tol"]
		self.overlap = config["overlap"]
		self.include_surroundings = config["include_surroundings"]
		if self.compute_stats:
			self.stats = {
				"chunk_size_dist": Counter(),
				"n_chunks_per_layout_dist": Counter(),
				"n_chunks_per_page_dist": Counter(),
				"n_chunks_per_doc_dist": Counter()
			}
		if self.compute_stats_examples:
			self.stats_examples = {key: {} for key in self.stats}

		assert self.chunk_size > 1, "chunk_size should be a non-negative non-zero integer."
		assert 0 <= self.chunk_size_tol <= 1, "chunk_size_tol should be a float between 0 and 1."
		assert self.overlap >= 0, "overlap should be a non-negative integer."
		assert self.overlap < self.chunk_size, "overlap should be less than chunk_size."

	def get_chunks(
			self,
			words: list, # (bs, n_pages, n_words)
			boxes: list, # (bs, n_pages, n_words, 4)
			layout_info: Optional[list] = None, # (bs, n_pages)
			**kwargs
	) -> tuple:
		"""
		Converts words and boxes to chunks
			:param words: list of words
			:param boxes: list of boxes
			:param layout_info: list of layout information
			:param chunk_size: size of the chunk in words
			:param overlap: overlap of words between chunks
		"""
		bs = len(words)
		question_id = kwargs.get("question_id", None)

		# Extract layout_boxes and layout_labels from layout_info
		if layout_info != [[]]:
			layout_boxes = [[layout_info[b][p]["boxes"] for p in range(len(layout_info[b]))] for b in range(bs)] # (bs, n_pages, n_boxes, 4)
			layout_labels = [[layout_info[b][p]["labels"] for p in range(len(layout_info[b]))] for b in range(bs)] # (bs, n_pages, n_boxes)
		else:
			layout_boxes = None
			layout_labels = None

		layout_labels_chunks = [] # (bs, n_chunks)
		page_indices = [] # (bs, n_chunks)
		words_text_chunks = [] # (bs, n_chunks, n_words)
		words_boxes_chunks = [] # (bs, n_chunks, n_words, 4)
		words_layout_labels_pages = [] # (bs, n_pages, n_words)

		def make_chunks(_words, _boxes, _p, words_lst, boxes_lst, p_lst) -> int:
			prev_chunk_size = 0
			n_chunks = 0
			for i in range(0, len(_words), self.chunk_size - self.overlap):
				chunk_words = _words[i:i + self.chunk_size]
				chunk_boxes = _boxes[i:i + self.chunk_size]
				this_chunk_size = len(chunk_words)
				if (
						i > 0 and 
						_p == p_lst[-1] and 
						prev_chunk_size + (this_chunk_size - self.overlap) <= self.chunk_size * (1+self.chunk_size_tol)
				): # if previous+this chunk in same page/layout is small, merge them
					this_chunk_size = prev_chunk_size + this_chunk_size - self.overlap
					words_lst[-1].extend(chunk_words[self.overlap:])
					boxes_lst[-1].extend(chunk_boxes[self.overlap:])
					self.stat_subtract("chunk_size_dist", prev_chunk_size)
					self.stat_sum("chunk_size_dist", this_chunk_size)
					self.stat_remove_example("chunk_size_dist", prev_chunk_size, f"{question_id[b]}_p{p}")
					self.stat_add_example("chunk_size_dist", this_chunk_size, f"{question_id[b]}_p{p}")
				else:
					p_lst.append(_p)
					words_lst.append(chunk_words)
					boxes_lst.append(chunk_boxes)
					self.stat_sum("chunk_size_dist", len(chunk_words))
					self.stat_add_example("chunk_size_dist", len(chunk_words), f"{question_id[b]}_p{p}")
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
					self.stat_sum("n_chunks_per_page_dist", page_n_chunks)
					self.stat_add_example("n_chunks_per_page_dist", page_n_chunks, f"{question_id[b]}_p{p}")
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
							contain_ratio = containment_ratio(box, layout_box)
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

						self.stat_sum("n_chunks_per_layout_dist", layout_n_chunks)
						self.stat_add_example("n_chunks_per_layout_dist", layout_n_chunks, f"{question_id[b]}_p{p}")
					batch_page_indices.extend([p] * len(layout_words_text))
					batch_words_text_chunks.extend(layout_words_text)
					batch_words_box_chunks.extend(layout_words_boxes)
					batch_words_layout_labels_pages.append(page_words_layout_labels)
					batch_n_chunks += page_n_chunks
					self.stat_sum("n_chunks_per_page_dist", page_n_chunks)
					self.stat_add_example("n_chunks_per_page_dist", page_n_chunks, f"{question_id[b]}_p{p}")

			layout_labels_chunks.append(batch_layout_labels_chunks)
			page_indices.append(batch_page_indices)
			words_text_chunks.append(batch_words_text_chunks)
			words_boxes_chunks.append(batch_words_box_chunks)
			words_layout_labels_pages.append(batch_words_layout_labels_pages)
			self.stat_sum("n_chunks_per_doc_dist", batch_n_chunks)
			self.stat_add_example("n_chunks_per_doc_dist", batch_n_chunks, f"{question_id[b]}")
		
		return (
			words_text_chunks, # (bs, n_chunks, n_words)
			words_boxes_chunks, # (bs, n_chunks, n_words, 4)
			layout_labels_chunks, # (bs, n_chunks)
			page_indices, # (bs, n_chunks)
			words_layout_labels_pages # (bs, n_pages, n_words)
		)
	
	@staticmethod
	def compact_chunks(
			words_text_chunks: list, # (bs, n_chunks, n_words)
			words_boxes_chunks: list # (bs, n_chunks, n_words, 4)
	) -> tuple:
		"""
		Converts words and word boxes to compact chunks
			:param words_text_chunks: list of words
			:param words_boxes_chunks: list of boxes
		"""
		text_chunks = [] # (bs, n_chunks)
		boxes_chunks = [] # (bs, n_chunks, 4)

		for b, (batch_words_text_chunks, batch_words_box_chunks) in enumerate(zip(words_text_chunks, words_boxes_chunks)):
			batch_text_chunks = []
			batch_box_chunks = []
			for chunk_words, chunk_boxes in zip(batch_words_text_chunks, batch_words_box_chunks):
				batch_text_chunks.append(" ".join(chunk_words))
				# Find box of the chunk
				min_x = min([box[0] for box in chunk_boxes])
				min_y = min([box[1] for box in chunk_boxes])
				max_x = max([box[2] for box in chunk_boxes])
				max_y = max([box[3] for box in chunk_boxes])
				batch_box_chunks.append([min_x, min_y, max_x, max_y])
			text_chunks.append(batch_text_chunks)
			boxes_chunks.append(batch_box_chunks)

		return text_chunks, boxes_chunks


class Embedder:
	def __init__(
			self,
			config: dict, 
			language_model: Optional[VT5ForConditionalGeneration]=None
	):
		# Load config
		self.embed_model = config.get("embed_model", "VT5")
		self.embed_weights = config.get("embed_weights", None)
		self.device = config.get("device", "cuda")
		self.cache_dir = config.get("cache_dir", None)
		self.language_model = language_model

		if self.embed_model == "VT5":
			self.embedding_dim = 768
			print(f"Using VT5 language backbone as embedding model")
		else:
			if self.embed_model == "BGE":
				if self.embed_weights is None:
					self.embed_weights = "BAAI/bge-small-en-v1.5"
				self.embedding_dim = 384
			elif self.embed_model == "BGE-M3":
				self.embed_weights = "BAAI/bge-m3"
				self.embedding_dim = 1024
			elif self.embed_model == "BGE-reranker":
				self.embed_weights = "BAAI/bge-reranker-v2-m3"
				self.embedding_dim = 1024
			self.bge_model = SentenceTransformer(self.embed_weights, cache_folder=self.cache_dir)
			print(f"Loading embedding model from {self.embed_weights}")

	def to(self, device: Any):
		if self.embed_model != "VT5":
			self.bge_model.to(device)

	def embed(self, text: List[str]) -> torch.Tensor:
		"""
		Embed a list of text
			:param text: list of strings
			:return: tensor of embeddings
		"""
		if not text:
			return torch.empty(0, self.embedding_dim).to(self.device)
		if self.embed_model == "VT5":
			input_ids, attention_mask = self.language_model.tokenizer(
				text,
				return_tensors="pt",
				padding=True,
				truncation=True
			).values()
			input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
			text_tokens_embeddings = self.language_model.language_backbone.shared(input_ids)
			text_embeddings = mean_pooling(text_tokens_embeddings, attention_mask)
		else:
			text_embeddings = self.bge_model.encode(text, convert_to_tensor=True)
		return text_embeddings

	def embed_multi(self, text: List[List[str]]) -> List[torch.Tensor]:
		"""
		Embed a list of lists of text
			:param text: list of lists of strings
			:return: list of tensors of embeddings
		"""
		return [self.embed(t) for t in text]


class Retriever(StatComponent):
	def __init__(self, config: dict):
		# Load config
		super(Retriever, self).__init__(config)
		self.k = config["n_chunks"]
		self.include_surroundings = config["include_surroundings"]
		if self.compute_stats:
			self.stats["layout_labels_topk_dist"] = {label: 0 for label in layout_map.values()}

	def _get_similarities(
			self,
			text_embeddings: List[torch.Tensor], # (bs, n_chunks, hidden_size)
			question_embeddings: torch.Tensor # (bs, hidden_size)
	) -> torch.Tensor:
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
		
		return similarities # (bs, n_chunks)

	def _get_top_k(
			self,
			similarities: List[torch.Tensor], # (bs, n_chunks)
			words_text_chunks: list, # (bs, n_chunks, n_words)
			words_box_chunks: list, # (bs, n_chunks, n_words, 4)
			layout_labels_chunks: list, # (bs, n_chunks)
			images: list, # (bs,) PIL images
			page_indices: list # (bs, n_chunks)
	):
		bs = len(similarities)
		top_k_words_text = []  # (bs, k, n_words)
		top_k_words_boxes = []  # (bs, k, n_words, 4)
		top_k_layout_labels = []  # (bs, k)
		top_k_page_indices = []  # (bs, k)

		for b in range(bs):
			k_min = min(self.k, len(similarities[b]))
			top_k = torch.topk(similarities[b], k=k_min, dim=-1).indices

			# these do not contain surrounding words, just the retrieved chunks
			top_k_layout_labels.append([layout_labels_chunks[b][i] for i in top_k])
			top_k_page_indices.append([page_indices[b][i] for i in top_k])

			# Include surrounding words
			# 	Build per-page word lists and mappings
			page_words_text = {}             # page_idx -> list of all words on that page
			page_words_boxes = {}            # page_idx -> list of all boxes on that page
			chunk_word_positions = {}        # page_idx -> {chunk_idx: (start_pos, end_pos)}
			included_word_indices = {}       # page_idx -> set of word indices already included
			total_chunks = len(top_k_layout_labels[b])

			# 	Build word lists and chunk positions per page
			for i in range(total_chunks):
				page_idx = page_indices[b][i]
				if page_idx not in page_words_text:
					page_words_text[page_idx] = []
					page_words_boxes[page_idx] = []
					chunk_word_positions[page_idx][i] = {}
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

			# 	Collect words with surroundings for each top-k chunk
			batch_top_k_words_text = []
			batch_top_k_words_boxes = []

			for i in top_k:
				i = i.item()
				page_idx = page_indices[b][i]
				(start_pos, end_pos) = chunk_word_positions[page_idx][i]

				# Determine the range of word indices to include
				surround_start = max(0, start_pos - self.include_surroundings)
				surround_end = min(len(page_words_text[page_idx]), end_pos + self.include_surroundings)

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

			# 	Append batch data to the final results
			top_k_words_text.append(batch_top_k_words_text) # (bs, k, n_words)
			top_k_words_boxes.append(batch_top_k_words_boxes)

		top_k_text, top_k_boxes = Chunker.compact_chunks(top_k_words_text, top_k_words_boxes)
		top_k_words_layout_labels = [
			[
				[top_k_layout_labels[b][i]] * len(top_k_words_text[b][i])
				for i in range(len(top_k_words_text[b]))
			]
			for b in range(bs)
		] # (bs, k, n_words)

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

		for b in range(bs):
			for c in range(len(top_k_layout_labels[b])):
				label = top_k_layout_labels[b][c]
				self.stat_sum("layout_labels_topk_dist", layout_map[label])
		
		return (
			top_k_text, # (bs, k)
			top_k_boxes, # (bs, k, 4)
			top_k_layout_labels, # (bs, k)
			top_k_words_text, # (bs, k, n_words)
			top_k_words_boxes, # (bs, k, n_words, 4)
			top_k_words_layout_labels, # (bs, k, n_words)
			top_k_patches, # (bs, k, h, w, 3)
			top_k_page_indices # (bs, k)
		)
	
	def retrieve(
			self,
			text_embeddings: List[torch.Tensor], # (bs, n_chunks, hidden_size)
			question_embeddings: torch.Tensor, # (bs, hidden_size)
			words_text_chunks: list, # (bs, n_chunks, n_words)
			words_box_chunks: list, # (bs, n_chunks, n_words, 4)
			layout_labels_chunks: list, # (bs, n_chunks)
			images: list, # (bs,) PIL images
			page_indices: list # (bs, n_chunks)
	) -> tuple:
		"""
		Retrieve the top-k chunks
			:param text_embeddings: list of text embeddings
			:param question_embeddings: question embeddings
			:param words_text_chunks: list of words
			:param words_box_chunks: list of boxes
			:param layout_labels_chunks: list of layout labels
			:param images: list of images
			:param page_indices: list of page indices
			:return: top-k chunks
		"""
		similarities = self._get_similarities(text_embeddings, question_embeddings)
		top_k = self._get_top_k(
			similarities, words_text_chunks, words_box_chunks, layout_labels_chunks, images, page_indices
		)
		return *top_k, similarities
