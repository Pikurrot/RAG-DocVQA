import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import gc
import logging
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from transformers import T5Config, AutoFeatureExtractor, AutoModel, AutoImageProcessor, BeitForSemanticSegmentation
from sentence_transformers import SentenceTransformer
from torch.nn import LayerNorm as BertLayerNorm, CrossEntropyLoss
from typing import Optional, Any, Dict, List, Tuple, Literal
from collections import Counter
from PIL import Image
from time import time
from abc import ABC, abstractmethod
from src.utils import containment_ratio, non_maximum_suppression
from src._model_utils import mean_pooling

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
        raise ValueError(f"Invalid layout model choice: {model_choice}")


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
	_layout_map = {
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

	def __init__(self, config: dict):
		torch.nn.Module.__init__(self)
		StatComponent.__init__(self, config)

		# Load config
		self.model_path = config.get("layout_model_weights", "cmarkea/dit-base-layout-detection")
		self.device = config["device"]
		self.cache_dir = config["cache_dir"]
		self.use_layout_labels = config["use_layout_labels"]
		self.layout_bs = config["layout_batch_size"]

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
		filtered_boxes, filtered_labels = [], []

		# Normalize boxes and compute weighted normalized areas
		normalized_boxes = [
			[box[0] / w, box[1] / h, box[2] / w, box[3] / h] for box in boxes
		]
		def weighted_area(nb):
			width = nb[2] - nb[0]
			height = nb[3] - nb[1]
			if height == 0:
				return 0
			return (width * height) * ((width / height) ** aspect_power)

		weighted_areas = [weighted_area(nb) for nb in normalized_boxes]
		params_to_show = []

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
				filtered_labels.append(labels[i])
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
			if self.use_layout_labels:
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
				labels_.extend([10]*len(bbx))
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
		self.device = config["device"]
		self.cache_dir = config["cache_dir"]
		self.use_layout_labels = config["use_layout_labels"]
		self.layout_bs = config["layout_batch_size"]

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
		self.chunk_size = config["chunk_size"]
		self.chunk_size_tol = config["chunk_size_tol"]
		self.overlap = config["overlap"]
		self.include_surroundings = config["include_surroundings"]
		self.page_retrieval = config["page_retrieval"]
		layout_map = get_layout_model_map(config)
		layout_map = {v: k for k, v in layout_map.items()}
		self.default_layout_label = layout_map["text"]
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
					page_layout_boxes = batch_layout_boxes[p]
					page_layout_labels = batch_layout_labels[p]
					layout_words_text = [] # (n_chunks, n_words)
					layout_words_boxes = [] # (n_chunks, n_words, 4)
					layout_indices = [] # (n_chunks,)
					page_n_chunks = 0
					page_words_layout_labels = [self.default_layout_label] * len(page_words) # (n_words,)
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


class Embedder:
	def __init__(
			self,
			config: dict, 
			language_model: Optional[Any]=None
	):
		# Load config
		self.embed_model = config.get("embed_model", "VT5")
		self.embed_weights = config.get("embed_weights", None)
		self.device = config.get("device", "cuda")
		self.cache_dir = config.get("cache_dir", None)
		self.language_model = language_model

		if self.embed_model == "VT5":
			self.embedding_dim = 768
			print("Using VT5 language backbone as embedding model")
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
		self.k = config["chunk_num"]
		self.include_surroundings = config["include_surroundings"]
		self.layout_map = get_layout_model_map(config)
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
				self.stat_sum("layout_labels_topk_dist", self.layout_map[label])
		
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
