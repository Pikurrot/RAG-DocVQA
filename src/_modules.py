import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import gc
import logging
import warnings
import networkx as nx
import math
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from transformers import T5Config, AutoFeatureExtractor, AutoModel, AutoImageProcessor, BeitForSemanticSegmentation
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder as sentence_transformers_CrossEncoder
from FlagEmbedding import FlagLLMReranker as FlagEmbedding_FlagLLMReranker
from torch.nn import LayerNorm as BertLayerNorm, CrossEntropyLoss
from typing import Optional, Any, Dict, List, Tuple, Literal, Union
from collections import Counter
from PIL import Image
from time import time
from abc import ABC, abstractmethod
from src.utils import containment_ratio, non_maximum_suppression, late_interaction, rectangles_overlap
from src._model_utils import mean_pooling
from src.custom_pix2struct_processor import extract_flattened_patches_single
from transformers.image_utils import infer_channel_dimension_format, to_numpy_array
from transformers.image_transforms import normalize

warnings.filterwarnings(
    "ignore", 
    message=".*not valid for `BeitImageProcessor.preprocess`.*", 
    category=UserWarning
)
logging.getLogger("doclayout_yolo").setLevel(logging.WARNING)

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
			config.visual_module_config["model_weights"],
			ignore_mismatched_sizes=True
		)
		self.image_model = AutoModel.from_pretrained(
			config.visual_module_config["model_weights"],
			ignore_mismatched_sizes=True
		)
		self.visual_emb_matcher = MLP(self.image_model.config.hidden_size, 0, self.image_model.config.hidden_size, 1)

		if not config.visual_module_config.get("finetune", False):
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


# For the Hi-VT5 model
class PageRetrievalModule(nn.Module):

	def __init__(self, config):
		super(PageRetrievalModule, self).__init__()

		self.page_retrieval = nn.Linear(config.max_doc_pages * config.page_tokens * config.hidden_size, config.max_doc_pages)
		# TODO Check if BinaryCrossEntropy allows to extend to longer sequences.

		if config.page_retrieval_config["loss"].lower() in ["ce", "crossentropy", "crossentropyloss"]:
			self.retrieval_criterion = CrossEntropyLoss()

		self.retrieval_loss_weight = config.page_retrieval_config["loss_weight"]

	def forward(self, document_embeddings, answer_page_idx):
		document_embeddings = document_embeddings.view([len(document_embeddings), -1])
		# document_embeddings = F.pad(document_embeddings, (0, self.page_retrieval.in_features-document_embeddings.shape[-1]), "constant", 0)  # In case is the last batch

		try:
			ret_logits = self.page_retrieval(document_embeddings)  # 10*2*512

		except:  # noqa: E722
			pad_document_embeddings = torch.zeros([len(document_embeddings), self.page_retrieval.in_features], dtype=document_embeddings.dtype, device=document_embeddings.device)
			pad_document_embeddings[:, :document_embeddings.shape[-1]] = document_embeddings
			ret_logits = self.page_retrieval(pad_document_embeddings.to())  # 10*2*512

		ret_loss = self.retrieval_criterion(ret_logits, answer_page_idx) * self.retrieval_loss_weight if answer_page_idx is not None else None

		return ret_loss, ret_logits


class StatComponent:
	def __init__(self, config: dict):
		self.compute_stats = config["compute_stats"]
		self.compute_stats_examples = config["compute_stats_examples"] and self.compute_stats
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


def get_layout_model_map(config: dict) -> dict:
    model_choice = config.get("layout_model")
    if model_choice == "YOLO":
        return LayoutModelYOLO.get_layout_map()
    elif model_choice == "DIT":
        return LayoutModelDIT.get_layout_map()
    else:
        return {1: "text"}

def get_raw_layout_model_map(config: dict) -> dict:
	model_choice = config.get("layout_model")
	if model_choice == "YOLO":
		return LayoutModelYOLO._layout_map_raw
	elif model_choice == "DIT":
		return LayoutModelDIT._layout_map_raw
	else:
		return {1: "text"}

class LayoutModelBase(torch.nn.Module, StatComponent, ABC):
	@property
	@abstractmethod
	def layout_map(self) -> dict:
		pass

	@classmethod
	@abstractmethod
	def get_layout_map(cls):
		raise 

	@abstractmethod
	def forward(
			self,
			images: List[Image.Image],
			return_steps: bool=False
	) -> Tuple[List[dict], List[dict]]:
		pass

	@abstractmethod
	def batch_forward(
			self,
			images: List[List[Image.Image]],
			return_steps: bool=False,
			**kwargs
	):
		pass


class LayoutModelDIT(LayoutModelBase):
	_layout_map_raw = {
		0: "Background",
		1: "Caption",
		2: "Footnote",
		3: "Formula",
		4: "List-item",
		5: "Page-footer",
		6: "Page-header",
		7: "Picture",
		8: "Section-header",
		9: "Table",
		10: "Text",
		11: "Title"
	}
	_layout_map = {
		0: "title",
		1: "text",
		2: "figure",
		3: "table"
	}

	def __init__(self, config: dict):
		torch.nn.Module.__init__(self)
		StatComponent.__init__(self, config)

		# Load config
		self.model_path = config.get("layout_model_weights", "cmarkea/dit-base-layout-detection")
		self.device = config.get("device", "cuda")
		self.cache_dir = config.get("cache_dir", None)
		self.use_layout_labels = config.get("use_layout_labels", "Default")
		self.layout_bs = config.get("layout_batch_size", 1)
		self.default_layout_label = {val: key for key, val in self.layout_map.items()}["text"]

		# Load layout model
		self.processor = AutoImageProcessor.from_pretrained(self.model_path, cache_dir=self.cache_dir)
		self.model = BeitForSemanticSegmentation.from_pretrained(self.model_path, cache_dir=self.cache_dir)
		self.model.to(self.device)

		# Stats
		if self.compute_stats:
			self.stats["n_layouts_per_page_dist"] = Counter()
			self.stats["layouts_size_w_dist"] = Counter()
			self.stats["layouts_size_h_dist"] = Counter()
			self.stats["layout_labels_dist"] = {LayoutModelDIT.layout_map[label]: 0 for label in LayoutModelDIT.layout_map}
			self.stats_examples["n_layouts_per_page_dist"] = {}
			self.stats_examples["layouts_size_w_dist"] = {}
			self.stats_examples["layouts_size_h_dist"] = {}

	@property
	def layout_map(self) -> dict:
		return LayoutModelDIT._layout_map

	@classmethod
	def get_layout_map(cls):
		return cls._layout_map

	def _filter_detections(
			self,
			boxes: List[List[int]],
			labels: List[int],
			image_size: Tuple[int, int],
			min_area: float=0.001,
			containment_threshold: float=0.5,
			condition: Literal["or", "and", "small", "overlap"]="or",
			aspect_power: float=1.0
	):
		"""
		Filters bounding boxes based on normalized size and overlap with condition.

		Args:
		- boxes: List of bounding boxes [xmin, ymin, xmax, ymax].
		- labels: List of labels corresponding to boxes.
		- image_size: Tuple (height, width) of the image.
		- min_area: Minimum normalized area (relative to image size) for a box to be valid.
		- containment_threshold: threshold to filter overlapping small boxes.
		- condition: "or", "and", "small", "overlap" for combining small area and overlap conditions.
		- aspect_power: Power to apply to the (width/height) aspect ratio.

		Returns:
		- Filtered boxes and labels.
		"""
		assert condition in {"or", "and", "small", "overlap"}, "Condition must be "or" or "and"."
		h, w = image_size

		# Map labels to relevant labels
		label_map = {
			0: None,
			1: 1,
			2: 1,
			3: None,
			4: 3,
			5: 1,
			6: 1,
			7: 2,
			8: 0,
			9: 3,
			10: 1,
			11: 0
		}
		relevant_boxes, relevant_labels = [], []
		for box, label in zip(boxes, labels):
			if label_map[label] is not None:
				relevant_boxes.append(box)
				relevant_labels.append(label_map[label])

		# Normalize boxes and compute weighted normalized areas
		normalized_boxes = [
			[box[0] / w, box[1] / h, box[2] / w, box[3] / h] for box in relevant_boxes
		]
		def weighted_area(nb):
			width = nb[2] - nb[0]
			height = nb[3] - nb[1]
			if height == 0:
				return 0
			return (width * height) * ((width / height) ** aspect_power)

		weighted_areas = [weighted_area(nb) for nb in normalized_boxes]
		params_to_show = []

		filtered_boxes, filtered_labels = [], []
		for i, box_a in enumerate(normalized_boxes):
			is_small = weighted_areas[i] < min_area
			is_overlapping = False
			max_cont = 0

			for j, box_b in enumerate(normalized_boxes):
				if i != j and weighted_areas[j] > weighted_areas[i]:  # Only compare to larger boxes
					ratio = containment_ratio(box_a, box_b)
					max_cont = max(max_cont, ratio)
					if ratio >= containment_threshold:
						is_overlapping = True
						break

			# Apply the condition ("and"/"or") for filtering
			if condition == "or":
				should_filter = is_small or is_overlapping
			elif condition == "and":
				should_filter = is_small and is_overlapping
			elif condition == "small":
				should_filter = is_small
			elif condition == "overlap":
				should_filter = is_overlapping

			if not should_filter:
				filtered_boxes.append(box_a)
				filtered_labels.append(relevant_labels[i])
				params_to_show.append(weighted_areas[i])

		# Denormalize the final boxes back to pixel coordinates
		denormalized_boxes = [
			[int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)]
			for box in filtered_boxes
		]
		return denormalized_boxes, filtered_labels

	def _detect_bboxes(self, masks: np.ndarray):
		"""
		A simple bounding box detection function
		"""
		detected_blocks = []
		contours, _ = cv2.findContours(
			masks.astype(np.uint8),
			cv2.RETR_TREE,
			cv2.CHAIN_APPROX_SIMPLE
		)
		for contour in list(contours):
			if len(list(contour)) >= 4:
				# smallest rectangle containing all points
				x, y, width, height = cv2.boundingRect(contour)
				bounding_box = [x, y, x + width, y + height]
				detected_blocks.append(bounding_box)
		return detected_blocks

	def forward(
			self,
			images: List[Image.Image],
			return_steps: bool=False
	) -> Tuple[List[dict], List[dict]]:
		with torch.inference_mode():
			# Preprocess images
			inputs = self.processor(images, return_tensors="pt", padding=False)
			inputs.to(self.device)

			# Forward pass
			outputs = self.model(**inputs)

			# Postprocess outputs
			segmentation = self.processor.post_process_semantic_segmentation(
				outputs,
				target_sizes=[image.size[::-1] for image in images]
			)
			segmentation = [seg.cpu() for seg in segmentation]

		del inputs, outputs
		gc.collect()
		torch.cuda.empty_cache()

		# Find bounding boxes
		bbox_pred = []
		for img_seg in segmentation:
			boxes_, labels_ = [], []
			mm = img_seg > 0
			if mm.sum() > 0:
				bbx = self._detect_bboxes(mm.numpy())
				boxes_.extend(bbx)
			if self.use_layout_labels != "Default":
				# Majority voting excluding class 0
				for box in boxes_:
					xmin, ymin, xmax, ymax = box
					segment_crop = img_seg[ymin:ymax, xmin:xmax]
					segment_crop = segment_crop[segment_crop != 0]
					if len(segment_crop) > 0:
						label = np.argmax(np.bincount(segment_crop.flatten()))
					else:
						label = 0
					labels_.append(label)
			else:
				labels_.extend([self.default_layout_label]*len(bbx))
			bbox_pred.append(dict(boxes=boxes_, labels=labels_))

		# Filter bounding boxes
		bbox_pred_filtered = []
		for i, (image, img_preds) in enumerate(zip(images, bbox_pred)):
			filtered_boxes, filtered_labels = self._filter_detections(
				img_preds["boxes"], img_preds["labels"], image.size[::-1],
				min_area=0.02, containment_threshold=0.6, condition = "or", aspect_power=1.0
			)
			# normalize boxes to be between 0 and 1
			filtered_boxes = [
				[box[0] / image.size[0], box[1] / image.size[1], box[2] / image.size[0], box[3] / image.size[1]]
				for box in filtered_boxes
			]
			bbox_pred_filtered.append(dict(boxes=filtered_boxes, labels=filtered_labels))

		if return_steps:
			steps = {
				"segmentation": segmentation,
				"layout_info_raw": bbox_pred
			}
		else:
			steps = {}

		return bbox_pred_filtered, steps

	def batch_forward(
			self,
			images: List[List[Image.Image]], # (bs, n_pages)
			return_steps: bool=False,
			**kwargs
	):
		"""
		Process a batch of images
		"""
		question_id = kwargs.get("question_id", None)
		bs = len(images)
		start_time = time()
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
			batch_layout_boxes, steps = self(batch_images, return_steps=True)
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
		
		if self.compute_stats:
			self.stats["layout_time"] = time() - start_time
			for b in range(bs):
				for p in range(len(layout_info[b])):
					# n_layouts_per_page_dist
					n_layouts = len(layout_info[b][p]["boxes"])
					self.stats["n_layouts_per_page_dist"][n_layouts] += 1
					if question_id:
						self.stat_add_example("n_layouts_per_page_dist", n_layouts, f"{question_id[b]}_p{p}")
					# layouts_size_w_dist, layouts_size_h_dist
					for box in layout_info[b][p]["boxes"]:
						w = box[2] - box[0]
						h = box[3] - box[1]
						self.stats["layouts_size_w_dist"][w] += 1
						self.stats["layouts_size_h_dist"][h] += 1
						if question_id:
							self.stat_add_example("layouts_size_w_dist", w, f"{question_id[b]}_p{p}")
							self.stat_add_example("layouts_size_h_dist", h, f"{question_id[b]}_p{p}")
					# layout_labels_dist
					for label in layout_info[b][p]["labels"]:
						self.stats["layout_labels_dist"][LayoutModelDIT.layout_map[label]] += 1
		
		if return_steps:
			steps = {
				"layout_segments": layout_segments, # (bs, n_pages, h, w)
				"layout_info_raw": layout_info_raw # (bs, n_pages)
			}
		else:
			steps = {}
		return layout_info, steps


class LayoutModelYOLO(LayoutModelBase):
	_layout_map_raw = {
		0: "title",
		1: "plain text",
		2: "abandon",
		3: "figure",
		4: "figure_caption",
		5: "table",
		6: "table_caption",
		7: "table_footnote",
		8: "isolate_formula",
		9: "formula_caption"
	}

	_layout_map = {
		0: "title",
		1: "text",
		2: "figure",
		3: "table"
	}

	def __init__(self, config: dict):
		torch.nn.Module.__init__(self)
		StatComponent.__init__(self, config)

		# Load config
		repo_id = config.get("layout_model_weights", "juliozhao/DocLayout-YOLO-DocStructBench")
		self.device = config.get("device", "cuda")
		self.cache_dir = config.get("cache_dir", None)
		self.use_layout_labels = config.get("use_layout_labels", "Default")
		self.layout_bs = config.get("layout_batch_size", 1)

		# Load layout model
		model_path = hf_hub_download(
			repo_id=repo_id,
			filename="doclayout_yolo_docstructbench_imgsz1024.pt",
			cache_dir=self.cache_dir
		)
		self.model = YOLOv10(model_path)
		self.model.to(self.device)

	@property
	def layout_map(self) -> dict:
		return LayoutModelYOLO._layout_map

	@classmethod
	def get_layout_map(cls):
		return cls._layout_map

	def _filter_detections(
			self,
			boxes: List[List[float]],
			labels: List[int],
			iou_threshold: float=0.7
	) -> Tuple[List[List[float]], List[int]]:
		"""
		Filters bounding boxes based label relevance and overlapping.

		Args:
		- boxes: List of bounding boxes [xmin, ymin, xmax, ymax].
		- labels: List of labels corresponding to boxes.

		Returns:
		- Filtered boxes and labels.
		"""
		# Map labels to relevant labels
		label_map = {
			0: 0,
			1: 1,
			2: 1,
			3: 2,
			4: 2,
			5: 3,
			6: 3,
			7: 3,
			8: None,
			9: None
		}
		filtered_boxes, filtered_labels = [], []
		for box, label in zip(boxes, labels):
			if label_map[label] is not None:
				filtered_boxes.append(box)
				filtered_labels.append(label_map[label])
		
		# Filter overlapping boxes (take biggest)
		keep_indices = non_maximum_suppression(filtered_boxes, iou_threshold=iou_threshold)
		filtered_boxes = [filtered_boxes[i] for i in keep_indices]
		filtered_labels = [filtered_labels[i] for i in keep_indices]
		
		return filtered_boxes, filtered_labels

	def forward(
			self,
			images: List[Image.Image],
			return_steps: bool=False
	) -> Tuple[List[dict], List[dict]]:
		# Forward pass
		det_res = self.model.predict(
			images,
			imgsz=1024,
			conf=0.2,
			device=self.device
		)

		# Find bounding boxes
		bbox_pred = []
		for res in det_res:
			boxes_, labels_ = [], []
			for box, label in zip(res.boxes.xyxyn, res.boxes.cls):
				boxes_.append(box.cpu().numpy().tolist()) # normalized
				labels_.append(label.item())
			bbox_pred.append(dict(boxes=boxes_, labels=labels_))
		del det_res
		gc.collect()
		torch.cuda.empty_cache()

		# Filter bounding boxes by label
		bbox_pred_filtered = []
		for i, (image, img_preds) in enumerate(zip(images, bbox_pred)):
			filtered_boxes, filtered_labels = self._filter_detections(
				img_preds["boxes"], img_preds["labels"]
			)
			bbox_pred_filtered.append(dict(boxes=filtered_boxes, labels=filtered_labels))

		if return_steps:
			steps = {
				"layout_info_raw": bbox_pred,
				"layout_segments": []
			}
		else:
			steps = {}
		return bbox_pred_filtered, steps
	
	def batch_forward(
			self,
			images: List[List[Image.Image]], # (bs, n_pages)
			return_steps: bool=False,
			**kwargs
	):
		"""
		Process a batch of images
		"""
		question_id = kwargs.get("question_id", None)
		bs = len(images)
		start_time = time()
		flatten_images = []
		for b in range(bs):
			page_images = images[b]
			flatten_images.extend(page_images)
		
		# Divide into batches of layout_bs
		new_batches = []
		for i in range(0, len(flatten_images), self.layout_bs):
			batch_images = flatten_images[i:i + self.layout_bs]
			new_batches.append(batch_images)

		# Process batches and flatten again
		flatten_layout_info = []
		flatten_layout_info_raw = []
		for batch_images in new_batches:
			batch_layout_boxes, steps = self(batch_images, return_steps=True)
			flatten_layout_info.extend(batch_layout_boxes)
			flatten_layout_info_raw.extend(steps["layout_info_raw"])

		# Reshape flatten_layout_boxes back to (bs, n_pages, n_boxes, 4)
		layout_info = []
		layout_info_raw = []
		index = 0
		for b in range(bs):
			page_layouts = []
			page_info_raw = []
			for p in range(len(images[b])):
				page_layouts.append(flatten_layout_info[index])
				page_info_raw.append(flatten_layout_info_raw[index])
				index += 1
			layout_info.append(page_layouts)
			layout_info_raw.append(page_info_raw)
		
		if self.compute_stats:
			self.stats["layout_time"] = time() - start_time
			for b in range(bs):
				for p in range(len(layout_info[b])):
					# n_layouts_per_page_dist
					n_layouts = len(layout_info[b][p]["boxes"])
					self.stats["n_layouts_per_page_dist"][n_layouts] += 1
					if question_id:
						self.stat_add_example("n_layouts_per_page_dist", n_layouts, f"{question_id[b]}_p{p}")
					# layouts_size_w_dist, layouts_size_h_dist
					for box in layout_info[b][p]["boxes"]:
						w = box[2] - box[0]
						h = box[3] - box[1]
						self.stats["layouts_size_w_dist"][w] += 1
						self.stats["layouts_size_h_dist"][h] += 1
						if question_id:
							self.stat_add_example("layouts_size_w_dist", w, f"{question_id[b]}_p{p}")
							self.stat_add_example("layouts_size_h_dist", h, f"{question_id[b]}_p{p}")
					# layout_labels_dist
					for label in layout_info[b][p]["labels"]:
						self.stats["layout_labels_dist"][LayoutModelYOLO.layout_map[label]] += 1

		if return_steps:
			steps = {
				"layout_info_raw": layout_info_raw,
				"layout_segments": [[]]
			}
		else:
			steps = {}
		return layout_info, steps


class LayoutModel(LayoutModelBase):
	def __new__(cls, config: dict):
		model_choice = config.get("layout_model")
		if model_choice == "YOLO":
			return LayoutModelYOLO(config)
		elif model_choice == "DIT":
			return LayoutModelDIT(config)
		else:
			raise ValueError(f"Invalid layout model choice: {model_choice}")


class Chunker(StatComponent):
	def __init__(self, config: dict):
		# Load config
		super(Chunker, self).__init__(config)
		self.chunk_size = config.get("chunk_size", 60)
		self.chunk_size_tol = config.get("chunk_size_tol", 0.2)
		self.overlap = config.get("overlap", 10)
		self.page_retrieval = config.get("page_retrieval", "concat")
		layout_map = get_layout_model_map(config)
		layout_map = {v: k for k, v in layout_map.items()}
		self.default_layout_label = layout_map["text"]
		self.cluster_layouts = config.get("cluster_layouts", False)

		if self.compute_stats:
			self.stats = {
				"chunk_size_dist": Counter(),
				"n_chunks_per_page_dist": Counter(),
				"n_chunks_per_doc_dist": Counter()
			}
			if not config["layout_model_weights"] or self.page_retrieval == "oracle":
				self.stats["n_chunks_per_layout_dist"] = Counter()

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
		layout_boxes = None
		layout_labels = None
		layout_clusters = None
		if layout_info != [[]]:
			layout_boxes = [[layout_info[b][p]["boxes"] for p in range(len(layout_info[b]))] for b in range(bs)] # (bs, n_pages, n_boxes, 4)
			layout_labels = [[layout_info[b][p]["labels"] for p in range(len(layout_info[b]))] for b in range(bs)] # (bs, n_pages, n_boxes)
			if "clusters" in layout_info[0][0].keys() and self.cluster_layouts:
				layout_clusters = [[layout_info[b][p]["clusters"] for p in range(len(layout_info[b]))] for b in range(bs)] # (bs, n_pages, n_boxes)

		layout_labels_chunks = [] # (bs, n_chunks)
		page_indices = [] # (bs, n_chunks)
		words_text_chunks = [] # (bs, n_chunks, n_words)
		words_boxes_chunks = [] # (bs, n_chunks, n_words, 4)
		words_layout_labels_pages = [] # (bs, n_pages, n_words)

		def make_chunks(
				_words: List[str],
				_boxes: List[list],
				_p: int,
				words_lst: list, # (n_chunks, n_words)
				boxes_lst: list, # (n_chunks, n_words, 4)
				p_lst: list # (n_chunks,)
		) -> int:
			
			prev_chunk_size = 0
			n_chunks = 0
			for i in range(0, len(_words), self.chunk_size - self.overlap):
				chunk_words = _words[i:i + self.chunk_size]
				chunk_boxes = _boxes[i:i + self.chunk_size]
				this_chunk_size = len(chunk_words)
				if (
						i > 0 and # not first chunk
						_p == p_lst[-1] and # same page
						prev_chunk_size + (this_chunk_size - self.overlap) <= self.chunk_size * (1+self.chunk_size_tol)
						# if previous+this chunk in same page/layout is small, merge them
				):
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
			batch_layout_boxes = None
			batch_layout_labels = None
			batch_layout_clusters = None
			if layout_boxes:
				batch_layout_boxes = layout_boxes[b]
				batch_layout_labels = layout_labels[b]
				if layout_clusters:
					batch_layout_clusters = layout_clusters[b]
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
				
				if self.page_retrieval == "oracle":
					# If oracle, take the whole page as a chunk
					batch_page_indices.append(p)
					batch_words_text_chunks.append(page_words)
					batch_words_box_chunks.append(page_boxes)
					batch_layout_labels_chunks.append(self.default_layout_label)
					batch_words_layout_labels_pages.append([self.default_layout_label] * len(page_words))
					batch_n_chunks += 1
					self.stat_sum("chunk_size_dist", len(page_words))
					self.stat_sum("n_chunks_per_page_dist", 1)
					self.stat_add_example("chunk_size_dist", len(page_words), f"{question_id[b]}_p{p}")
					self.stat_add_example("n_chunks_per_page_dist", 1, f"{question_id[b]}_p{p}")
					continue

				if batch_layout_boxes is None or len(batch_layout_boxes[p]) == 0:
					# If no layout, make chunks inside the page
					page_n_chunks = make_chunks(
						page_words, page_boxes, p,
						batch_words_text_chunks, batch_words_box_chunks, batch_page_indices
					)
					batch_layout_labels_chunks.extend([self.default_layout_label] * page_n_chunks)
					batch_words_layout_labels_pages.append([self.default_layout_label] * len(page_words))
					batch_n_chunks += page_n_chunks
					self.stat_sum("n_chunks_per_page_dist", page_n_chunks)
					self.stat_add_example("n_chunks_per_page_dist", page_n_chunks, f"{question_id[b]}_p{p}")
				else:
					# Else, if layout, make chunks inside the layout boxes
					page_layout_boxes = batch_layout_boxes[p] # (n_layouts, 4)
					page_layout_labels = batch_layout_labels[p] # (n_layouts,)
					if batch_layout_clusters:
						page_layout_clusters = batch_layout_clusters[p].tolist() # (n_layouts,)
					else:
						page_layout_clusters = None
					layout_words_text = [] # (n_chunks, n_words)
					layout_words_boxes = [] # (n_chunks, n_words, 4)
					layout_indices = [] # (n_chunks,) Not used, just for consistency
					page_n_chunks = 0
					page_words_layout_labels = [self.default_layout_label] * len(page_words) # (n_words,)
					# Sort layout boxes left-right and top-bottom
					if batch_layout_boxes is not None and len(batch_layout_boxes[p]) > 0:
						if batch_layout_clusters:
							sorted_tuples = sorted(
								zip(page_layout_boxes, page_layout_labels, page_layout_clusters),
								key=lambda x: (x[0][0], x[0][1])  # sort by xmin, then by ymin
							)
							page_layout_boxes, page_layout_labels, page_layout_clusters = map(list, zip(*sorted_tuples))
						else:
							sorted_tuples = sorted(
								zip(page_layout_boxes, page_layout_labels),
								key=lambda x: (x[0][0], x[0][1])
							)
							page_layout_boxes, page_layout_labels = map(list, zip(*sorted_tuples))			
					# Find words inside the layout box
					layout_words_inside = [] # (n_layouts, n_words)
					layout_boxes_inside = [] # (n_layouts, n_words, 4)
					layout_labels_inside = page_layout_labels.copy() # (n_layouts,)
					for lb, (layout_box, layout_label) in enumerate(zip(page_layout_boxes, page_layout_labels)):
						words_inside = []
						boxes_inside = []
						for i, (word, box) in enumerate(zip(page_words, page_boxes)):
							contain_ratio = containment_ratio(box, layout_box)
							if contain_ratio > 0.5:
								words_inside.append(word)
								boxes_inside.append(box)
								page_words_layout_labels[i] = layout_label
						layout_words_inside.append(words_inside)
						layout_boxes_inside.append(boxes_inside)
					# If clusters provided, concatenate words and boxes of layout clusters
					if page_layout_clusters:
						cluster_words_inside = [] # (n_clusters, n_words)
						cluster_boxes_inside = [] # (n_clusters, n_words, 4)
						cluster_layout_labels = [] # (n_clusters,)
						cluster2idx = {}
						c = 0
						# group by cluster
						for lb, (words_inside, boxes_inside, layout_label, cluster) in enumerate(
							zip(layout_words_inside, layout_boxes_inside, page_layout_labels, page_layout_clusters)
						):
							if cluster == -1:  # treat each -1 as a separate cluster
								cluster_words_inside.append(words_inside)
								cluster_boxes_inside.append(boxes_inside)
								cluster_layout_labels.append(Counter([layout_label]))
								c += 1
							else:
								if cluster not in cluster2idx:
									cluster2idx[cluster] = c
									cluster_words_inside.append(words_inside)
									cluster_boxes_inside.append(boxes_inside)
									cluster_layout_labels.append(Counter([layout_label]))
									c += 1
								else:
									idx = cluster2idx[cluster]
									cluster_words_inside[idx].extend(words_inside)
									cluster_boxes_inside[idx].extend(boxes_inside)
									cluster_layout_labels[idx][layout_label] += 1
						layout_words_inside = cluster_words_inside # (n_clusters, n_words)
						layout_boxes_inside = cluster_boxes_inside # (n_clusters, n_words, 4)
						layout_labels_inside = [cluster_labels.most_common(1)[0][0] for cluster_labels in cluster_layout_labels] # (n_clusters,)
					# Split the words inside the layout box into chunks
					for lb, (words_inside, boxes_inside, layout_label) in enumerate(
						zip(layout_words_inside, layout_boxes_inside, layout_labels_inside)
					):
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
				try:
					min_x = min([box[0] for box in chunk_boxes])
					min_y = min([box[1] for box in chunk_boxes])
					max_x = max([box[2] for box in chunk_boxes])
					max_y = max([box[3] for box in chunk_boxes])
				except ValueError:
					min_x, min_y, max_x, max_y = 0, 0, 1, 1
				batch_box_chunks.append([min_x, min_y, max_x, max_y])
			text_chunks.append(batch_text_chunks)
			boxes_chunks.append(batch_box_chunks)

		return text_chunks, boxes_chunks


class ImageChunker:
	def __init__(self, config: dict):
		# Load config
		self.patch_size = config.get("patch_size", 256)
		self.overlap = config.get("overlap", 0)
		self.mode = config.get("chunk_mode", "square")
		layout_map = get_layout_model_map(config)
		layout_map = {v: k for k, v in layout_map.items()}
		self.default_layout_label = layout_map["text"]
		self.cluster_layouts = config.get("cluster_layouts", False)
	
	def divide_image_into_patches(
			self,
			image: Image.Image,
	):
		patch_size = self.patch_size
		overlap = patch_size // 2 if self.overlap else 0
		mode = self.mode

		width, height = image.size
		patches = []
		patches_matrix = []
		patches_xyxy = []
		assert mode in ["square", "horizontal", "page"]

		# Determine step size
		step_size = patch_size - overlap
		
		if mode == "page":
			# Page mode - use the entire image as a single patch
			patches.append(image)
			patches_matrix = [[image]]  # Single element matrix
			patches_xyxy.append([0, 0, width, height])
		
		elif mode == "square":
			# Original square patch implementation
			num_patches_w = math.ceil((width - overlap) / step_size)
			num_patches_h = math.ceil((height - overlap) / step_size)
			
			# Initialize the 2D matrix to store patches
			patches_matrix = [[None for _ in range(num_patches_w)] for _ in range(num_patches_h)]
			
			for i in range(num_patches_h):
				for j in range(num_patches_w):
					# Calculate the top-left coordinate for the patch
					left = j * step_size
					top = i * step_size
					
					# Ensure the patch does not exceed the image dimensions
					right = min(left + patch_size, width)
					bottom = min(top + patch_size, height)
					
					# Adjust the starting coordinate if the patch is smaller than patch_size
					if right - left < patch_size:
						left = max(right - patch_size, 0)
					if bottom - top < patch_size:
						top = max(bottom - patch_size, 0)
						
					patch = image.crop((left, top, right, bottom))
					patches.append(patch)
					patches_matrix[i][j] = patch
					patches_xyxy.append([left, top, right, bottom])
					
		elif mode == "horizontal":
			# Horizontal mode - full width strips with fixed height
			num_patches_h = math.ceil((height - overlap) / step_size)
			
			# For horizontal mode, we only have one column
			patches_matrix = [[None] for _ in range(num_patches_h)]
			
			# Check if we need to handle a special case for the last patch
			last_patch_height = height - (num_patches_h - 1) * step_size
			if 0 < last_patch_height < patch_size and num_patches_h > 1:
				# We'll create one fewer patch and extend the last one
				actual_num_patches = num_patches_h - 1
			else:
				actual_num_patches = num_patches_h
				
			for i in range(actual_num_patches):
				# Calculate the top-left coordinate for the patch
				left = 0  # Always start from the left edge
				
				if i == actual_num_patches - 1 and actual_num_patches < num_patches_h:
					# This is the last patch and we're handling the special case
					# Make it extend to the bottom of the image
					top = i * step_size
					bottom = height
				else:
					top = i * step_size
					bottom = min(top + patch_size, height)
				
				# Always use the full width
				right = width
				
				# For all but the last patch in the special case, keep the standard sizing
				if i < actual_num_patches - 1 or actual_num_patches == num_patches_h:
					if bottom - top < patch_size:
						top = max(bottom - patch_size, 0)
						
				patch = image.crop((left, top, right, bottom))
				patches.append(patch)
				patches_matrix[i][0] = patch
				patches_xyxy.append([left, top, right, bottom])
			
			# If we reduced the number of patches, resize the matrix
			if actual_num_patches < num_patches_h:
				patches_matrix = patches_matrix[:actual_num_patches]
		
		# Return the patches and the matrix (without converting to numpy array)
		return patches, patches_matrix, patches_xyxy

	def crop_boxes(
			self,
			image: Image.Image,
			boxes: List[list],
			labels: List[int],
			clusters: Optional[List[int]] = None
	) -> Tuple[List[Image.Image], List[int]]:
		crops = []
		clustered_boxes = []
		clustered_labels = []
		
		if clusters is None:
			clustered_boxes = boxes
			clustered_labels = labels
		else:
			# Group boxes by cluster. Redefines box coordinates so that all fit
			box_clusters = {}
			label_clusters = {}
			for i, (box, label, cluster) in enumerate(zip(boxes, labels, clusters)):
				if cluster == -1:
					clustered_boxes.append(box)  
					clustered_labels.append(label)
				else:
					if cluster not in box_clusters:
						box_clusters[cluster] = []
						label_clusters[cluster] = []
					box_clusters[cluster].append(boxes[i])
					label_clusters[cluster].append(labels[i])
			
			for cluster in box_clusters.keys():
				min_x = min([box[0] for box in box_clusters[cluster]])
				min_y = min([box[1] for box in box_clusters[cluster]])
				max_x = max([box[2] for box in box_clusters[cluster]])
				max_y = max([box[3] for box in box_clusters[cluster]])
				clustered_boxes.append([min_x, min_y, max_x, max_y])
				
				# Take label with largest total area in cluster
				cluster_labels = label_clusters[cluster]
				label_areas = {}
				for label in set(cluster_labels):
					# Sum areas of all boxes with this label
					total_area = sum([
						(box[2] - box[0]) * (box[3] - box[1])  # width * height
						for box, l in zip(box_clusters[cluster], cluster_labels) 
						if l == label
					])
					label_areas[label] = total_area
				most_common_label = max(label_areas.items(), key=lambda x: x[1])[0]  
				clustered_labels.append(most_common_label)

		for box in clustered_boxes:
			box = box.copy()
			box[0] = int(box[0] * image.width)
			box[1] = int(box[1] * image.height) 
			box[2] = int(box[2] * image.width)
			box[3] = int(box[3] * image.height)
			cropped_image = image.crop(box)
			crops.append(cropped_image)

		return crops, clustered_labels
	
	def get_chunks(
			self,
			images: list, # (bs, n_pages)
			layout_info: Optional[list] = None, # (bs, n_pages)
			**kwargs
	) -> tuple:
		bs = len(images)

		# Extract layout_boxes and layout_labels from layout_info
		layout_boxes = None
		layout_labels = None
		layout_clusters = None
		if layout_info != [[]]:
			layout_boxes = [[layout_info[b][p]["boxes"] for p in range(len(layout_info[b]))] for b in range(bs)] # (bs, n_pages, n_boxes, 4)
			layout_labels = [[layout_info[b][p]["labels"] for p in range(len(layout_info[b]))] for b in range(bs)] # (bs, n_pages, n_boxes)
			if "clusters" in layout_info[0][0].keys() and self.cluster_layouts:
				layout_clusters = [[layout_info[b][p]["clusters"] for p in range(len(layout_info[b]))] for b in range(bs)] # (bs, n_pages, n_boxes)

		patches_flatten = []
		patches_flatten_indices = []
		patches_matrix_list = []
		patches_xyxy = []
		for b in range(bs):
			batch_patches_flatten = []
			batch_patches_flatten_indices = []
			batch_patches_matrix_list = []
			batch_patches_xyxy = []
			patch_count = 0
			batch_layout_boxes = None
			batch_layout_labels = None
			batch_layout_clusters = None
			if layout_boxes:
				batch_layout_boxes = layout_boxes[b]
				batch_layout_labels = layout_labels[b]
				if layout_clusters:
					batch_layout_clusters = layout_clusters[b]
			for p in range(len(images[b])):
				if batch_layout_boxes is None or len(batch_layout_boxes[p]) == 0:
					# If no layout, make chunks inside the page
					box_patches, box_patches_matrix, box_patches_xyxy = self.divide_image_into_patches(images[b][p])
					if len(box_patches) == 0:
						continue
					batch_patches_flatten.extend(box_patches)
					batch_patches_flatten_indices.extend([patch_count] * len(box_patches))
					batch_patches_matrix_list.append(box_patches_matrix)
					batch_patches_xyxy.append(box_patches_xyxy)
					patch_count += 1
				else:
					# Else, if layout, make chunks inside the layout boxes
					page_layout_boxes = batch_layout_boxes[p]
					page_layout_labels = batch_layout_labels[p]
					if batch_layout_clusters:
						page_layout_clusters = batch_layout_clusters[p].tolist() # (n_layouts,)
						# Sort layout boxes left-right and top-bottom
						sorted_tuples = sorted(
							zip(page_layout_boxes, page_layout_labels, page_layout_clusters),
							key=lambda x: (x[0][0], x[0][1])
						)
						page_layout_boxes, page_layout_labels, page_layout_clusters = map(list, zip(*sorted_tuples))
					else:
						page_layout_clusters = None
						# Sort layout boxes left-right and top-bottom
						sorted_tuples = sorted(
							zip(page_layout_boxes, page_layout_labels),
							key=lambda x: (x[0][0], x[0][1])
						)
						page_layout_boxes, page_layout_labels = map(list, zip(*sorted_tuples))
					# Crop boxes in the image
					page_cropped_layout_boxes, page_cropped_layout_labels = self.crop_boxes(
						images[b][p], page_layout_boxes, page_layout_labels, page_layout_clusters
					)
					# Divide the cropped boxes into patches
					for i in range(len(page_cropped_layout_boxes)):
						# if layout label = 1, divide, else keep the whole box
						if page_cropped_layout_labels[i] == 1:
							box_patches, box_patches_matrix, box_patches_xyxy = self.divide_image_into_patches(page_cropped_layout_boxes[i])
							if len(box_patches) == 0: # box was probably too small
								continue
						else:
							# for images and tables do not divide
							box_patches = [page_cropped_layout_boxes[i]]
							box_patches_matrix = np.array([[page_cropped_layout_boxes[i]]])
							box_patches_xyxy = np.array([[0, 0, page_cropped_layout_boxes[i].width, page_cropped_layout_boxes[i].height]])
						batch_patches_flatten.extend(box_patches)
						batch_patches_flatten_indices.extend([patch_count] * len(box_patches))
						batch_patches_matrix_list.append(box_patches_matrix)
						batch_patches_xyxy.append(box_patches_xyxy)
						patch_count += 1
			patches_flatten.append(batch_patches_flatten)
			patches_flatten_indices.append(np.array(batch_patches_flatten_indices))
			patches_matrix_list.append(batch_patches_matrix_list)
			patches_xyxy.append(batch_patches_xyxy)
		return patches_flatten, patches_flatten_indices, patches_matrix_list, patches_xyxy


class BaseEmbedder(ABC):
	def __init__(
			self,
			config: dict
	):
		self.device = config.get("device", "cuda")
		self.cache_dir = config.get("cache_dir", None)
		self.embedding_dim = None

	@abstractmethod
	def forward(self, text: Any):
		raise NotImplementedError
	
	def batch_forward(self, text: Any):
		return [self.forward(t) for t in text]
	
	def __call__(self, text: Any):
		return self.forward(text)


class BiEncoder(BaseEmbedder):
	def __init__(
			self,
			config: dict, 
			language_model: Optional[Any]=None
	):
		# Load config
		super(BiEncoder, self).__init__(config)
		self.embed_weights = config.get("embed_weights", None)
		self.embed_model = config.get("embed_model", "BGE")
		self.language_model = language_model
		self.device = config.get("device", "cuda")

		if self.embed_model == "VT5":
			print("Using VT5 language backbone as embedding model")
		elif self.embed_model == "BGE":
			self.model = SentenceTransformer(self.embed_weights, cache_folder=self.cache_dir, device=self.device)
			print(f"Loading embedding model from {self.embed_weights}")
		elif self.embed_model == "JINA":
			self.model = SentenceTransformer(self.embed_weights, cache_folder=self.cache_dir, trust_remote_code=True, device=self.device)
			self.model.max_seq_length = 1024
		self.embedding_dim = self.get_embedding_dim()

	def get_embedding_dim(self):
		if self.embed_model == "VT5":
			return 768
		elif self.embed_model == "BGE":
			return self.model[1].word_embedding_dimension
		elif self.embed_model == "JINA":
			return self.model.get_sentence_embedding_dimension()
		
	def to(self, device):
		if self.embed_model in ["BGE", "JINA"]:
			self.model.to(device)

	def forward(self, text: List[str]) -> torch.Tensor:
		"""
		Embed a list of text
			:param text: list of strings
			:return: tensor of embeddings
		"""
		if not text:
			return torch.empty(0, self.embedding_dim).to(self.device)
		if self.embed_weights == "VT5":
			input_ids, attention_mask = self.language_model.tokenizer(
				text,
				return_tensors="pt",
				padding=True,
				truncation=True
			).values() # (bs, seq_len), (bs, seq_len)
			input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
			text_tokens_embeddings = self.language_model.language_backbone.shared(input_ids) # (bs, seq_len, hidden_dim)
			text_embeddings = mean_pooling(text_tokens_embeddings, attention_mask) # (bs, hidden_dim)
		else:
			text_embeddings = self.model.encode(text, convert_to_tensor=True) # (bs, hidden_dim)
		return text_embeddings


class CrossEncoder(BaseEmbedder):
	def __init__(
			self,
			config: dict
	):
		super(CrossEncoder, self).__init__(config)
		self.reranker_weights = config.get("reranker_weights", None)
		self.reranker_model = config.get("reranker_model", "BGE")
		self.device = config.get("device", "cuda")

		self.model = sentence_transformers_CrossEncoder(self.reranker_weights, cache_dir=self.cache_dir, device=self.device)
		self.embedding_dim = self.get_embedding_dim()
		print(f"Loading reranker model from {self.reranker_weights}")

	def get_embedding_dim(self):
		if self.reranker_model == "BGE":
			return self.model.model.roberta.encoder.layer[-1].output.dense.out_features
		
	def to(self, device):
		if self.reranker_model == "BGE":
			self.model.model.to(device)

	def forward(self, text: List[Tuple[str, str]]) -> np.ndarray:
		"""
		Compute the scores of a list of text pairs
			:param text: list of pairs of strings
			:return: array of probabilities
		"""
		if not text:
			return torch.zeros(len(text)).to(self.device)
		return self.model.predict(text)


class FlagLLMReranker(BaseEmbedder):
	def __init__(
			self,
			config: dict
	):
		super(FlagLLMReranker, self).__init__(config)
		self.reranker_weights = config.get("reranker_weights", None)
		self.reranker_model = config.get("reranker_model", "BGE")
		self.device = config.get("device", "cuda")

		self.model = FlagEmbedding_FlagLLMReranker(self.reranker_weights, cache_dir=self.cache_dir, use_fp16=True, device=self.device)
		self.embedding_dim = self.get_embedding_dim()
		print(f"Loading reranker model from {self.reranker_weights}")

	def get_embedding_dim(self):
		return self.model.model.base_model.norm.weight.shape.numel()
	
	def to(self, device):
		self.model.model.to(device)

	def forward(self, text: List[Tuple[str, str]]) -> np.ndarray:
		"""
		Compute the scores of a list of text pairs
			:param text: list of pairs of strings
			:return: array of probabilities
		"""
		if not text:
			return torch.zeros(len(text)).to(self.device)
		return self.model.compute_score(text)


class Reranker:
	def __init__(
			self,
			config: dict,
			cross_encoder: Optional[CrossEncoder]=None
	):
		# Load config
		self.rerank_filter_tresh = float(config.get("rerank_filter_tresh", 0.4))
		self.rerank_max_chunk_num = config.get("rerank_max_chunk_num", 5)
		self.rerank_min_chunk_num = config.get("rerank_min_chunk_num", 1)
		if cross_encoder is None:
			if "gemma" in config.get("reranker_weights", ""):
				self.cross_encoder = FlagLLMReranker(config)
			else:
				self.cross_encoder = CrossEncoder(config)
		else:
			self.cross_encoder = cross_encoder

	def rerank(
			self,
			question: str,
			candidates: List[str], # (k,)
			*args: List[Any] # (k, *)
	) -> tuple:
		"""
		Rerank a list of candidates given a question
			:param question: question string
			:param candidates: list of candidate strings
			:param args: additional arguments to rerank in the same order
			:return: reranked candidates and arguments
		"""
		question = [question] * len(candidates)
		pairs = list(zip(question, candidates))
		
		with torch.no_grad():
			scores = self.cross_encoder.forward(pairs)
		if isinstance(scores, torch.Tensor):
			scores = scores.cpu().numpy()
		sorted_indices = np.argsort(scores)[::-1]

		# Filter candidates
		filtered_indices = [i for i in sorted_indices if scores[i] >= self.rerank_filter_tresh]
		if len(filtered_indices) > self.rerank_max_chunk_num:
			filtered_indices = filtered_indices[:self.rerank_max_chunk_num]
		elif len(filtered_indices) < self.rerank_min_chunk_num:
			filtered_indices = sorted_indices[:self.rerank_min_chunk_num]
		sorted_indices = filtered_indices

		# Sort candidates and arguments
		sorted_candidates = [candidates[i] for i in sorted_indices]
		sorted_args = [[arg[i] for i in sorted_indices] for arg in args]
		return sorted_candidates, *sorted_args

	def batch_rerank(
			self,
			questions: List[str], # (bs, k)
			candidates: List[List[str]], # (bs, k)
			*args: List[List[Any]] # (bs, k, *)
	) -> tuple:
		sorted_candidates = [] # (bs, k)
		sorted_args = [[] for _ in range(len(args))] # (bs, k, *)
		for b, (question, batch_candidates, *batch_args) in enumerate(zip(questions, candidates, *args)):
			batch_sorted_candidates, *batch_sorted_args = self.rerank(question, batch_candidates, *batch_args)
			sorted_candidates.append(batch_sorted_candidates)
			for i in range(len(args)):
				sorted_args[i].append(batch_sorted_args[i])
		return sorted_candidates, *sorted_args


class ImageEncoder(BaseEmbedder):
	def __init__(
			self,
			config: dict,
			visual_encoder: Any
	):
		# Load config
		super(ImageEncoder, self).__init__(config)
		self.visual_encoder = visual_encoder
		self.embedder_batch_size = config.get("embedder_batch_size", 1)
		
	def to(self, device):
		self.visual_encoder.to(device)

	def forward(self, images: List[Image.Image]) -> torch.Tensor:
		batch_size = self.embedder_batch_size 
		if not isinstance(images, list):
			images = [images]
		inputs = []
		start = time()
		for i in range(len(images)):
			img = to_numpy_array(images[i])
			input_data_format = infer_channel_dimension_format(img)
			mean = np.mean(img)
			std = np.std(img)
			adjusted_stddev = max(std, 1.0 / math.sqrt(np.prod(img.shape)))
			img = normalize(
				img,
				mean=mean,
				std=adjusted_stddev,
				data_format=None,
				input_data_format=input_data_format
			)
			patches = extract_flattened_patches_single(
				img,
				max_patches=2048,
				patch_size={"height": 16, "width": 16},
				input_data_format=input_data_format,
				row_offset=0
			)[0]
			inputs.append(patches)
		inputs = np.array(inputs)
		inputs = torch.from_numpy(inputs).to(self.device)
		start = time()
		num_batches = (inputs.size(0) +batch_size - 1) // batch_size
		encoder_outputs = []
		for i in range(num_batches):
			batch_inputs = inputs[i * batch_size:(i + 1) * batch_size]
			batch_outputs = self.visual_encoder(batch_inputs)
			encoder_outputs.append(batch_outputs["last_hidden_state"])
		if len(encoder_outputs) == 0:
			return torch.zeros(1, 2048, 768).to(self.device)
		encoder_outputs = torch.cat(encoder_outputs, dim=0)
		return encoder_outputs


class S2Chunker:
	def __init__(
			self,
			config: dict,
			embedder: Optional[BiEncoder]=None
	):
		self.config = config
		self.cluster_mode = config.get("cluster_mode", "spatial+semantic")
		self.calculate_n_clusters = config.get("calculate_n_clusters", "heuristic")
		# self.max_token_length = config["chunk_size"] * (1+config["chunk_size_tol"])
		self.graph = None
		if self.cluster_mode == "spatial+semantic":
			if embedder is None:
				self.embedder = BiEncoder(config)
			else:
				self.embedder = embedder
				self.tokenizer = self.embedder.bge_model.tokenizer

	def create_nodes_and_edges(
			self,
			page_layout_info: Dict,
			page_info: Optional[Dict] = None
	) -> Tuple[List[Dict], List[Tuple]]:
		page_layout_boxes = page_layout_info["boxes"]
		page_layout_labels = page_layout_info["labels"]

		nodes = []
		edges = []
		used = np.zeros(len(page_layout_boxes), dtype=bool)
		i = 0

		# If in spatial mode or no page_info, just use layout boxes
		if self.cluster_mode == "spatial" or page_info is None:
			for l, (layout_box, layout_label) in enumerate(zip(page_layout_boxes, page_layout_labels)):
				node = {
					"global_id": i,
					"page": 1,
					"bbox": layout_box,
					"text": "",  # Empty text for spatial-only
					"label": layout_label,
				}
				nodes.append(node)
				i += 1
				used[l] = True
		else:
			# Original logic for spatial+semantic mode
			page_words = page_info["ocr_tokens"]
			page_boxes = page_info["ocr_normalized_boxes"]

			layout_words_text = []
			layout_words_boxes = []
			for lb, (layout_box, layout_label) in enumerate(zip(page_layout_boxes, page_layout_labels)):
				# Find words inside the layout box
				words_inside = []
				boxes_inside = []
				for i, (word, box) in enumerate(zip(page_words, page_boxes)):
					contain_ratio = containment_ratio(box, layout_box)
					if isinstance(box, np.ndarray):
						box = box.tolist()
					if contain_ratio > 0.5:
						words_inside.append(word)
						boxes_inside.append(box)
				layout_words_text.append(words_inside)
				layout_words_boxes.append(boxes_inside)

			for l, (layout_box, layout_label) in enumerate(zip(page_layout_boxes, page_layout_labels)):
				if not layout_words_text[l]:
					continue
				node = {
					"global_id": i,
					"page": 1,
					"bbox": layout_box,
					"text": " ".join(layout_words_text[l]),
					"label": layout_label,
				}
				nodes.append(node)
				i += 1
				used[l] = True

		# Add edges between all nodes
		for i in range(len(nodes)):
			for j in range(i + 1, len(nodes)):
				edges.append((nodes[i]['global_id'], nodes[j]['global_id']))
		
		return nodes, edges, used

	def _spatial_weights_calculation(
			self,
			nodes: List[Dict]
	) -> np.ndarray:
		try:
			num_nodes = len(nodes)
			spatial_weights = np.zeros((num_nodes, num_nodes))
			for i in range(num_nodes):
				for j in range(num_nodes):
					bbox_i = nodes[i]['bbox']
					bbox_j = nodes[j]['bbox']
					centroid_i = np.array([(bbox_i[0] + bbox_i[2]) / 2, (bbox_i[1] + bbox_i[3]) / 2])
					centroid_j = np.array([(bbox_j[0] + bbox_j[2]) / 2, (bbox_j[1] + bbox_j[3]) / 2])
					distance = np.linalg.norm(centroid_i - centroid_j)
					spatial_weights[i, j] = 1 / (1 + distance)
			return spatial_weights
		except Exception as e:
			print(f"Error in spatial_weights_calculation: {e}")
			return None

	def _semantic_weights_calculation(
			self,
			nodes: List[Dict]
	) -> np.ndarray:
		try:
			texts = [node['text'] for node in nodes if node.get('text', '').strip()]

			with torch.no_grad():
				embeddings = self.embedder.forward(texts).cpu().numpy()
			semantic_weights = cosine_similarity(embeddings)
			return semantic_weights
		except Exception as e:
			print(f"Error in semantic_weights_calculation: {e}")
			return None

	def _combined_weights(
			self,
			nodes: List[Dict]
	) -> np.ndarray:
		spatial_weights = self._spatial_weights_calculation(nodes)
		if self.cluster_mode == "spatial+semantic":
			semantic_weights = self._semantic_weights_calculation(nodes)
		else:
			semantic_weights = spatial_weights
		if spatial_weights is None or semantic_weights is None:
			raise ValueError("Spatial or semantic weight calculation failed.")
		combined_weights = (spatial_weights + semantic_weights) / 2
		return combined_weights

	def _create_graph(self, nodes: List[int], edges: List[tuple]) -> nx.Graph:
		graph = nx.Graph()
		graph.add_nodes_from(nodes)
		graph.add_edges_from(edges)
		return graph

	def _add_weights_to_graph(self, graph: nx.Graph, weights: np.ndarray) -> nx.Graph:
		for i, (u, v) in enumerate(graph.edges()):
			graph[u][v]['weight'] = weights[u, v]
		return graph

	def _calculate_n_clusters(
		self,
		nodes: List[Dict],
		weights: np.ndarray,
		min_k: int = 2,
		max_k: int = 10
	) -> int:
		# Compute degree and normalized Laplacian
		degree = np.sum(weights, axis=1)
		D_inv_sqrt = np.diag(1.0 / (np.sqrt(degree) + 1e-10))
		L_norm = np.eye(weights.shape[0]) - D_inv_sqrt @ weights @ D_inv_sqrt

		# Compute eigenvalues and eigenvectors
		eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
		# Use the smallest 'max_k' eigenvectors for spectral embedding
		embedding = eigenvectors[:, :max_k]

		best_k = min_k
		best_score = -1
		best_labels = np.full(len(nodes), -1)
		# Try different k values and choose the one with the highest silhouette score
		upper_bound = min(max_k, len(nodes) - 1) # k < n_samples or error
		for k in range(min_k, upper_bound + 1):
			if self.calculate_n_clusters == "heuristic":
				kmeans = KMeans(n_clusters=k, random_state=0).fit(embedding)
				labels = kmeans.labels_
			elif self.calculate_n_clusters == "best":
				clustering = SpectralClustering(n_clusters=k, affinity='precomputed')
				labels = clustering.fit_predict(weights)
			score = silhouette_score(embedding, labels)
			if score > best_score:
				best_score = score
				best_k = k
				best_labels = labels
		return best_k, best_labels

	def _cluster_graph(
			self,
			graph: nx.Graph,
			weights: np.ndarray,
			n_clusters: int = 3
	) -> Dict[int, int]:
		clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
		labels = clustering.fit_predict(weights)
		return {node: label for node, label in zip(graph.nodes(), labels)}

	def _group_nodes_by_cluster(self, clusters: Dict[int, int]) -> Dict[int, List[int]]:
		cluster_groups = {}
		for node, cluster_id in clusters.items():
			if cluster_id not in cluster_groups:
				cluster_groups[cluster_id] = []
			cluster_groups[cluster_id].append(node)
		return cluster_groups

	def _split_clusters_by_token_length(
			self,
			clusters: Dict[int, int],
			nodes: List[Dict]
	) -> Dict[int, int]:
		updated_clusters = {}
		cluster_id_counter = 0

		for cluster_id, node_ids in self._group_nodes_by_cluster(clusters).items():
			current_chunk = []
			current_token_length = 0

			for node_id in node_ids:
				node = next((n for n in nodes if n['global_id'] == node_id), None)
				if not node:
					continue
				node_text = node['text']
				node_token_length = len(self.tokenizer.tokenize(node_text))

				if current_token_length + node_token_length > self.max_token_length:
					for node_in_chunk in current_chunk:
						updated_clusters[node_in_chunk] = cluster_id_counter
					cluster_id_counter += 1
					current_chunk = []
					current_token_length = 0

				current_chunk.append(node_id)
				current_token_length += node_token_length

			for node_in_chunk in current_chunk:
				updated_clusters[node_in_chunk] = cluster_id_counter
			cluster_id_counter += 1

		return updated_clusters

	def cluster(
			self,
			nodes: List[Dict],
			edges: List[tuple]
	) -> Dict[int, int]:
		node_ids = [node['global_id'] for node in nodes]
		# print(node_ids, edges)
		graph = self._create_graph(node_ids, edges)
		weights = self._combined_weights(nodes)
		if weights is None:
			print("Weight calculation failed. Returning empty clusters.")
			return {}

		weighted_graph = self._add_weights_to_graph(graph, weights)
		self.graph = weighted_graph

		n_clusters, best_labels = self._calculate_n_clusters(nodes, weights)
		if self.calculate_n_clusters == "heuristic":
			clusters = self._cluster_graph(weighted_graph, weights, n_clusters)
			clusters = self._split_clusters_by_token_length(clusters, nodes)
		else:
			clusters = {node: label for node, label in zip(weighted_graph.nodes(), best_labels)}

		return clusters

	def forward(
			self,
			layout_info: List[Dict],
			pages_info: Optional[List[Dict]] = None
	) -> List[Dict]:
		batch_clusters = []
		for p, page_layout_info in enumerate(layout_info):
			try:
				if len(page_layout_info["boxes"]) == 0:
					batch_clusters.append(np.array([]))
					continue
					
				# Get corresponding page_info or None for spatial-only mode
				page_info = pages_info[p] if pages_info is not None else None
				
				nodes, edges, used = self.create_nodes_and_edges(page_layout_info, page_info)
				if len(nodes) < 2:
					batch_clusters.append(np.full(len(page_layout_info["boxes"]), -1))
					continue
					
				clusters = self.cluster(nodes, edges)
				clusters = sorted(clusters.items(), key=lambda item: item[0])
				clusters = [x[1] for x in clusters]
				complete_clusters = np.full(len(used), -1)
				complete_clusters[used] = clusters
				batch_clusters.append(complete_clusters)
			except Exception as e:
				print("p: ", p)
				print("page_layout_info:", page_layout_info)
				if pages_info is not None:
					print("page_info:", pages_info[p])
				raise e
		return batch_clusters

	def __call__(self, *args, **kwds):
		return self.forward(*args, **kwds)


class Retriever(StatComponent):
	def __init__(self, config: dict):
		# Load config
		super(Retriever, self).__init__(config)
		self.k = config.get("chunk_num", 10)
		self.include_surroundings = config.get("include_surroundings", 0)
		self.layout_map = get_layout_model_map(config)
		self.reorder_chunks = config.get("reorder_chunks", False)
		if self.compute_stats:
			self.stats["layout_labels_topk_dist"] = {label: 0 for label in self.layout_map.values()}

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

			total_chunks = len(similarities[b])

			# 	Build word lists and chunk positions per page
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

			# 	Collect words with surroundings for each top-k chunk
			batch_top_k_words_text = []
			batch_top_k_words_boxes = []

			for i in top_k:
				i = i.item()
				page_idx = page_indices[b][i]
				try:
					(start_pos, end_pos) = chunk_word_positions[page_idx][i]
				except KeyError as e:
					print(f"Page index: {page_idx}, Chunk index: {i}")
					print(f"Chunk word positions: {chunk_word_positions}")
					raise e

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
				page: Image.Image = images[b][page_idx]  # (H, W, 3)
				box: np.ndarray = top_k_boxes[b][i].copy()  # (4,)
				# transform to absolute coordinates
				box[0] = int(box[0] * page.width)
				box[1] = int(box[1] * page.height)
				box[2] = int(box[2] * page.width)
				box[3] = int(box[3] * page.height)
				# Ensure proper order of coordinates
				xmin = min(box[0], box[2])
				ymin = min(box[1], box[3])
				xmax = max(box[0], box[2])
				ymax = max(box[1], box[3])
				patch = page.crop([xmin, ymin, xmax, ymax])  # (h, w, 3)
				batch_patches.append(patch)
			top_k_patches.append(batch_patches)

		for b in range(bs):
			for c in range(len(top_k_layout_labels[b])):
				label = top_k_layout_labels[b][c]
				# self.stat_sum("layout_labels_topk_dist", self.layout_map[label])
		
		# Reorder chunks by page and top-left first
		if self.reorder_chunks:
			for b in range(bs):
				indices = sorted(
					range(len(top_k_page_indices[b])),
					key=lambda i: (top_k_page_indices[b][i], top_k_boxes[b][i][1], top_k_boxes[b][i][0])
				)
				top_k_text[b] = [top_k_text[b][i] for i in indices]
				top_k_boxes[b] = [top_k_boxes[b][i] for i in indices]
				top_k_layout_labels[b] = [top_k_layout_labels[b][i] for i in indices]
				top_k_words_text[b] = [top_k_words_text[b][i] for i in indices]
				top_k_words_boxes[b] = [top_k_words_boxes[b][i] for i in indices]
				top_k_words_layout_labels[b] = [top_k_words_layout_labels[b][i] for i in indices]
				top_k_patches[b] = [top_k_patches[b][i] for i in indices]
				top_k_page_indices[b] = [top_k_page_indices[b][i] for i in indices]

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


class VisualRetriever:
	def __init__(self, config: dict):
		# Load config
		self.k = config.get("chunk_num", 10)
		self.include_surroundings = config.get("include_surroundings", 0)
		self.mode = config.get("chunk_mode", "horizontal")
		self.layout_map = get_layout_model_map(config)

	def _get_similarities(
			self,
			patch_embeddings: List[torch.Tensor], # (bs, n_patches, seq_len, dim)
			question_embeddings: torch.Tensor # (bs, seq_len, hidden_size)
	) -> torch.Tensor:
		similarities = []
		bs = len(patch_embeddings)

		for i in range(bs):
			image_embeds_i = patch_embeddings[i] # (n_patches, seq_len, dim)
			question_embed_i = question_embeddings[i].unsqueeze(0) # (1, seq_len, hidden_size)
			scores = late_interaction(question_embed_i, image_embeds_i) # (n_patches,)
			similarities.append(scores)
		
		return similarities # (bs, n_patches)

	def _get_surrounding_patches(
			self,
			patch_coord: tuple, # (2,)
			patches_matrix: list, # (h, w)
			include_surroundings: Union[int, tuple]=0
	):
		"""
		Get patch coordinates in a specific pattern around a center patch.
		
		If include_surroundings is an integer:
			Pattern follows the sequence:
			0: Single center patch
			1: Add horizontal neighbors (3 patches)
			2: Add vertical neighbors to form a cross (5 patches)
			3: Complete square (9 patches)
			4: Add horizontal extensions (15 patches)
			5: Add vertical extensions (21 patches)
			6: Complete larger square (25 patches)
			... and continues with this pattern
		
		If include_surroundings is a tuple (x, y):
			Returns a rectangle of (2*y+1, 2*x+1) patches centered at patch_coord,
			where x is horizontal radius and y is vertical radius.
		"""
		row, col = patch_coord
		max_rows = len(patches_matrix)
		max_cols = len(patches_matrix[0]) if max_rows > 0 else 0
		coords = set()
		
		# Check if include_surroundings is a tuple
		if isinstance(include_surroundings, tuple) and len(include_surroundings) == 2:
			x_radius, y_radius = include_surroundings
			
			# Add all patches in the rectangle
			for r in range(row - y_radius, row + y_radius + 1):
				for c in range(col - x_radius, col + x_radius + 1):
					coords.add((r, c))
		else:
			# Original pattern-based logic for integer include_surroundings
			square_level = include_surroundings // 3
			phase = include_surroundings % 3
			
			# Calculate the current size of the completed square
			current_radius = square_level  # 0, 1, 2, 3, ...
			
			# Step 1: Add the completed square for the current level
			for r in range(row - current_radius, row + current_radius + 1):
				for c in range(col - current_radius, col + current_radius + 1):
					coords.add((r, c))
			
			# Step 2: Handle the phase-specific extensions
			if phase > 0:
				# Phase 1: Add horizontal extensions
				next_radius = current_radius + 1
				
				# Add horizontal extensions (left and right columns)
				for r in range(row - current_radius, row + current_radius + 1):
					coords.add((r, col - next_radius))  # Left column
					coords.add((r, col + next_radius))  # Right column
			
			if phase > 1:
				# Phase 2: Add vertical extensions
				next_radius = current_radius + 1
				
				# Add vertical extensions (top and bottom rows)
				for c in range(col - current_radius, col + current_radius + 1):
					coords.add((row - next_radius, c))  # Top row
					coords.add((row + next_radius, c))  # Bottom row
		
		# Filter out coordinates that are outside the matrix bounds
		valid_coords = []
		for r, c in coords:
			if 0 <= r < max_rows and 0 <= c < max_cols:
				valid_coords.append((r, c))
		
		return valid_coords

	def _merge_overlapping_patches(
			self,
			surrounding_coords_set: list, # List of tuples (page_idx, row, col)
			patches_matrix_list: list, # (bs, n_patches, h, w)
			patches_xyxy: list, # (bs, n_patches, 4)
			images: List[Image.Image] # (bs,) PIL images
	) -> list:
		"""
		Create non-overlapping patches by merging overlapping ones.
		
		Args:
			surrounding_coords_set: List of tuples (page_idx, row, col)
			patches_matrix_list: List of 2D matrices of patches (PIL images)
			patches_xyxy: List of lists containing patch coordinates [x1, y1, x2, y2]
			images: List of PIL images, each representing a page
			
		Returns:
			List of non-overlapping PIL Image patches
		"""
		# Group patches by page
		patches_by_page = {}
		for page_idx, row, col in surrounding_coords_set:
			if page_idx not in patches_by_page:
				patches_by_page[page_idx] = []
			
			# Make sure the indices are valid
			if (0 <= row < len(patches_matrix_list[page_idx]) and 
				0 <= col < len(patches_matrix_list[page_idx][0])):
				# Get the patch coordinates
				try:
					patch_xyxy = patches_xyxy[page_idx][row]
					# Check if patch_xyxy is actually a list of coordinates
					if isinstance(patch_xyxy, list) and len(patch_xyxy) == 4:
						patches_by_page[page_idx].append((patch_xyxy, (row, col)))
					else:
						# Try to find the correct format
						if isinstance(patches_xyxy[page_idx], list):
							# Try different access patterns
							if col < len(patches_xyxy[page_idx]):
								patch_xyxy = patches_xyxy[page_idx][col]
								if isinstance(patch_xyxy, list) and len(patch_xyxy) == 4:
									patches_by_page[page_idx].append((patch_xyxy, (row, col)))
				except (IndexError, TypeError) as e:
					print(f"Error accessing patches_xyxy[{page_idx}][{row}][{col}]: {e}")
					print(f"Shape of patches_xyxy[{page_idx}]: {len(patches_xyxy[page_idx])}")
					continue
		
		# Process each page separately
		merged_patches = []
		
		for page_idx, patches in patches_by_page.items():
			if not patches:
				continue
				
			# Build graph where nodes are patches and edges connect overlapping patches
			graph = {}
			for i in range(len(patches)):
				graph[i] = []
				xyxy_i, _ = patches[i]
				for j in range(len(patches)):
					if i != j:
						xyxy_j, _ = patches[j]
						if rectangles_overlap(xyxy_i, xyxy_j):
							graph[i].append(j)
			
			# Find connected components (clusters of overlapping patches)
			visited = [False] * len(patches)
			clusters = []
			
			for i in range(len(patches)):
				if not visited[i]:
					# Start a new cluster
					cluster = []
					queue = [i]
					visited[i] = True
					
					while queue:
						node = queue.pop(0)
						cluster.append(patches[node])
						
						for neighbor in graph[node]:
							if not visited[neighbor]:
								visited[neighbor] = True
								queue.append(neighbor)
					
					clusters.append(cluster)
			
			# Create a merged patch for each cluster
			for cluster in clusters:
				# Find the bounding box that encompasses all patches in the cluster
				xyxy_list = [xyxy for xyxy, _ in cluster]
				min_x = min(xyxy[0] for xyxy in xyxy_list)
				min_y = min(xyxy[1] for xyxy in xyxy_list)
				max_x = max(xyxy[2] for xyxy in xyxy_list)
				max_y = max(xyxy[3] for xyxy in xyxy_list)
				
				# Crop this region from the original page image
				crop = images[page_idx].crop((min_x, min_y, max_x, max_y))
				merged_patches.append(crop)
		
		return merged_patches

	def _get_top_k(
			self,
			similarities: List[torch.Tensor], # (bs, n_patches)
			patches_flatten_indices: list, # (bs, n_patches)
			patches_matrix_list: list, # (bs, n_patches, h, w)
			patches_xyxy: list, # (bs, n_patches, 4)
			images: List[List[Image.Image]] # (bs, n_pages)
	):
		bs = len(similarities)
		surrounding_patches_list = [] # (bs, k) PIL images
		page_indices_list = [] # (bs, k) page indices for each retrieved patch

		# Get coordinates of the patches
		for b in range(bs):
			batch_patch_coords = [] # (k, 3)
			batch_patches_flatten_indices = patches_flatten_indices[b] # (n_patches,)
			batch_patches_xyxy = patches_xyxy[b] # (n_patches, 4)
			if len(batch_patches_flatten_indices) == 0:
				surrounding_patches_list.append([])
				page_indices_list.append([])
				continue
			batch_patches_matrix_list = patches_matrix_list[b] # (n_patches, h, w)
			top_k = torch.topk(similarities[b], min(self.k, len(similarities[b]))).indices # (k,)
			top_k = top_k.cpu().numpy()
			for idx in top_k:
				page_idx = int(batch_patches_flatten_indices[idx])  # Convert to Python int
				idx_converted = idx - sum(np.count_nonzero(batch_patches_flatten_indices==i) for i in range(page_idx))
				if self.mode == "square":
					raise NotImplementedError()
				elif self.mode == "horizontal":
					row = int(idx_converted)  # Convert to Python int
					col = 0
					batch_patch_coords.append((page_idx, row, col))
		
			# Include surrounding patches
			surrounding_coords_set = set()
			for coord in batch_patch_coords:
				page_idx, row, col = coord
				patches_matrix = batch_patches_matrix_list[page_idx]
				surrounding_coords = self._get_surrounding_patches((row, col), patches_matrix, self.include_surroundings)
				surrounding_coords = [(page_idx, r, c) for r, c in surrounding_coords]
				surrounding_coords_set.update(surrounding_coords)
			surrounding_coords_set = list(surrounding_coords_set)

			# batch_surrounding_patches_list = []
			# for coord in surrounding_coords_set:
			# 	page_idx, row, col = coord
			# 	patches_matrix = batch_patches_matrix_list[page_idx]
			# 	if 0 <= row < len(patches_matrix) and 0 <= col < len(patches_matrix[0]):
			# 		batch_surrounding_patches_list.append(patches_matrix[row][col])
			# batch_surrounding_patches_list = [Image.fromarray(patch) for patch in batch_surrounding_patches_list]
			batch_surrounding_patches_list = self._merge_overlapping_patches(
				surrounding_coords_set,
				batch_patches_matrix_list,
				batch_patches_xyxy,
				images[b]
			)
			
			# Extract page indices from the surrounding coords and convert to Python integers
			batch_page_indices = [int(coord[0]) for coord in set((coord[0],) for coord in surrounding_coords_set)]
			
			surrounding_patches_list.append(batch_surrounding_patches_list)
			page_indices_list.append(batch_page_indices)

		return surrounding_patches_list, page_indices_list


	def retrieve(
			self,
			patch_embeddings: List[torch.Tensor], # (bs, n_patches, seq_len, dim)
			question_embeddings: torch.Tensor, # (bs, seq_len, hidden_size)
			patches_flatten_indices: list, # (bs, n_patches)
			patches_matrix_list: list, # (bs, n_patches, h, w)
			patches_xyxy: list, # (bs, n_patches, 4)
			images: List[List[Image.Image]] # (bs, n_pages)
	) -> tuple:
		similarities = self._get_similarities(patch_embeddings, question_embeddings)
		top_k, page_indices = self._get_top_k(similarities, patches_flatten_indices, patches_matrix_list, patches_xyxy, images)
		return top_k, page_indices


class NotAnswerableClassifier(nn.Module):
	"""
	A MLP that takes question, chunks and answer embeddings and outputs a probability of the answer being not answerable.
	"""
	def __init__(self, config: dict):
		super().__init__()
		self.config = config
		emb_dim = config["emb_dim"]
		self.input_dim = emb_dim * 2
		self.hidden_dim = config["hidden_dim"]
		self.num_layers = config["num_layers"]
		self.mlp = MLP(self.input_dim, self.hidden_dim, 1, self.num_layers)

	def to(self, device):
		self.mlp.to(device)

	def forward(
			self,
			input_embeddings: torch.Tensor, # (bs, seq_len, emb_dim)
			answer_embeddings: torch.Tensor, # (bs, seq_len2, emb_dim)
	):
		input_embeddings_mean = input_embeddings.mean(dim=1) # (bs, emb_dim)
		answer_embeddings_mean = answer_embeddings.mean(dim=1) # (bs, emb_dim)
		embeddings = torch.cat([input_embeddings_mean, answer_embeddings_mean], dim=1) # (bs, input_dim)
		out = self.mlp(embeddings) # (bs, 1)
		out = torch.sigmoid(out)
		return out # (bs, 1)
	
	def update_results(
			self,
			result: dict,
			input_embeddings: torch.Tensor,
			answer_embeddings: torch.Tensor
	) -> tuple:
		probs = self.forward(input_embeddings, answer_embeddings)
		pred_answers = result[1] # (bs, 1)
		pred_answers_conf = result[3] # (bs, 1)
		not_answerable = probs > 0.5
		pred_answers[not_answerable] = ""
		pred_answers_conf[not_answerable] = 0.0
		return (result[0], pred_answers, result[2], pred_answers_conf), probs
