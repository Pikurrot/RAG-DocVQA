import torch
import gc
from transformers import AutoImageProcessor, BeitForSemanticSegmentation
from typing import List, Tuple
from PIL import Image
import numpy as np
import cv2

layout_map = {
	0: 'Background',
	1: 'Caption',
	2: 'Footnote',
	3: 'Formula',
	4:'List-item',
	5: 'Page-footer',
	6: 'Page-header',
	7:'Picture',
	8: 'Section-header',
	9: 'Table',
	10: 'Text',
	11: 'Title'
}

class LayoutModel(torch.nn.Module):
	def __init__(self, config: dict):
		super(LayoutModel, self).__init__()

		# Load config
		self.model_path = config.get("layout_model_weights", "cmarkea/dit-base-layout-detection")
		self.device = config.get("device", "cuda")
		self.cache_dir = config.get("cache_dir", None)
		self.distinguish_labels = config.get("use_layout_labels", False)

		# Load layout model
		self.processor = AutoImageProcessor.from_pretrained(self.model_path, cache_dir=self.cache_dir)
		self.model = BeitForSemanticSegmentation.from_pretrained(self.model_path, cache_dir=self.cache_dir)
		self.model.to(self.device)

	def _containment_ratio(
			self,
			small_box: List[int],
			large_box: List[int]
	) -> float:
		"""Calculate the containment ratio of small_box in large_box."""
		x1 = max(small_box[0], large_box[0])
		y1 = max(small_box[1], large_box[1])
		x2 = min(small_box[2], large_box[2])
		y2 = min(small_box[3], large_box[3])
		inter_width = max(0, x2 - x1)
		inter_height = max(0, y2 - y1)
		inter_area = inter_width * inter_height
		small_area = (small_box[2] - small_box[0]) * (small_box[3] - small_box[1])
		return inter_area / small_area if small_area > 0 else 0

	def _filter_detections(
			self,
			boxes: List[List[int]],
			labels: List[int],
			image_size: Tuple[int, int],
			min_area: float=0.001,
			containment_threshold: float=0.5,
			condition: str="or",
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
		assert condition in {"or", "and", "small", "overlap"}, "Condition must be 'or' or 'and'."
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
					ratio = self._containment_ratio(box_a, box_b)
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
			if self.distinguish_labels:
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
			steps = None

		return bbox_pred_filtered, steps
