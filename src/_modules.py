import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config
from transformers import AutoFeatureExtractor, AutoModel
from torch.nn import LayerNorm as BertLayerNorm
from typing import Optional, Any, Dict
from collections import Counter
from src.utils import containment_ratio

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
	
	def compact_chunks(
			self,
			words_text_chunks: list, # (bs, n_pages, n_words)
			words_boxes_chunks: list # (bs, n_pages, n_words, 4)
	) -> tuple:
		"""
		Converts words and word boxes to compact chunks
			:param words_text_chunks: list of words
			:param words_boxes_chunks: list of boxes
		"""
		text_chunks = [] # (bs, n_chunks)
		boxes_chunks = [] # (bs, n_chunks, 4)

		for b, (batch_words, batch_boxes) in enumerate(zip(words_text_chunks, words_boxes_chunks)): # (n_pages, n_words), (n_pages, n_words, 4)
			batch_text_chunks = [] # (n_chunks,)
			batch_box_chunks = [] # (n_chunks, 4)
			for p, (page_words, page_boxes) in enumerate(zip(batch_words, batch_boxes)):
				# Join words and boxes for each chunk
				for chunk_words, chunk_boxes in zip(page_words, page_boxes):
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


class ChunkEmbedder:
	pass


class Retriever:
	pass
