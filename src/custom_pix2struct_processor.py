import math
import io
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import T5TokenizerFast  # minimal import for tokenizer
from transformers.utils import logging
from transformers.models.pix2struct.image_processing_pix2struct import (
	torch_extract_patches,
	render_header
)
from transformers.image_transforms import (
	to_channel_dimension_format,
	normalize, convert_to_rgb
)
from transformers.image_utils import (
	ChannelDimension,
	get_image_size,
	make_list_of_images,
	valid_images,
	to_numpy_array,
	infer_channel_dimension_format
)

logger = logging.get_logger(__name__)

# ----------------------------
# Custom Patch Extraction Functions
# ----------------------------

def extract_flattened_patches_single(
	image: np.ndarray,
	max_patches: int,
	patch_size: Dict[str, int],
	input_data_format: Optional[Union[str, str]] = None,
	row_offset: int = 0
) -> np.ndarray:
	"""
	Process a single image: resize to limit number of patches, extract patches (non-overlapping),
	prepend row and column IDs to each flattened patch, and pad/truncate to exactly max_patches.
	"""
	# Convert image to channels-first using our helper.
	image = to_channel_dimension_format(image, ChannelDimension.FIRST, input_data_format)
	image = torch.from_numpy(image)

	patch_height, patch_width = patch_size["height"], patch_size["width"]
	image_height, image_width = get_image_size(image, ChannelDimension.FIRST)

	# Compute scaling factor so that the number of patches roughly does not exceed max_patches.
	scale = math.sqrt(max_patches * (patch_height / image_height) * (patch_width / image_width))
	num_feasible_rows = max(min(math.floor(scale * image_height / patch_height), max_patches), 1)
	num_feasible_cols = max(min(math.floor(scale * image_width / patch_width), max_patches), 1)
	resized_height = max(num_feasible_rows * patch_height, 1)
	resized_width = max(num_feasible_cols * patch_width, 1)

	image = torch.nn.functional.interpolate(
		image.unsqueeze(0),
		size=(resized_height, resized_width),
		mode="bilinear",
		align_corners=False,
		antialias=True,
	).squeeze(0)

	# [1, rows, columns, patch_height * patch_width * image_channels]
	patches = torch_extract_patches(image, patch_height, patch_width)

	patches_shape = patches.shape
	rows = patches_shape[1]
	cols = patches_shape[2]
	depth = patches_shape[3]
	
	patches = patches.reshape([rows * cols, depth])
	
	# Generate row and column indices.
	row_ids = torch.arange(rows).reshape(rows, 1).repeat(1, cols).reshape(rows * cols, 1)
	col_ids = torch.arange(cols).reshape(1, cols).repeat(rows, 1).reshape(rows * cols, 1)
	row_ids += 1 + row_offset  # Offset to avoid zeros for padding + offset since one image is after the other
	col_ids += 1
	row_ids = row_ids.to(torch.float32)
	col_ids = col_ids.to(torch.float32)
	
	result = torch.cat([row_ids, col_ids, patches], dim=-1)  # Each patch becomes [row, col, features...]
	
	num_extracted = result.shape[0]
	if num_extracted < max_patches:
		pad_amt = max_patches - num_extracted
		padding = torch.zeros((pad_amt, result.shape[-1]), dtype=result.dtype)
		result = torch.cat([result, padding], dim=0)
	else:
		result = result[:max_patches]
	
	return to_numpy_array(result), int(row_ids.max().item())

def extract_multi_image_flattened_patches(
	images: List[np.ndarray],
	max_total_patches: int,
	patch_size: Dict[str, int],
	input_data_format: Optional[Union[str, str]] = None,
) -> np.ndarray:
	"""
	Process a list of images. Allocate an equal share of patches per image,
	concatenate them, and pad/truncate to exactly max_total_patches.
	"""
	num_images = len(images)
	if num_images == 0:
		raise ValueError("No images provided.")
	max_per_image = max_total_patches // num_images
	all_patches = []
	row_offset = 0
	for i, img in enumerate(images):
		patches, row_offset = extract_flattened_patches_single(
			image=img,
			max_patches=max_per_image,
			patch_size=patch_size,
			input_data_format=input_data_format,
			row_offset=row_offset
		)
		all_patches.append(patches)
	concatenated = np.concatenate(all_patches, axis=0)
	total = concatenated.shape[0]
	feature_dim = concatenated.shape[1]
	if total < max_total_patches:
		pad_amt = max_total_patches - total
		padding = np.zeros((pad_amt, feature_dim), dtype=concatenated.dtype)
		concatenated = np.concatenate([concatenated, padding], axis=0)
	else:
		concatenated = concatenated[:max_total_patches]
	return concatenated

# ----------------------------
# Custom Image Processor
# ----------------------------

class CustomPix2StructImageProcessor:
	"""
	A minimal custom image processor for Pix2Struct that:
	- Optionally converts images to RGB
	- Optionally normalizes images
	- Extracts flattened mini-patches from one or more images,
		preserving row and column indices
	- Limits the total number of patches via a max_total_patches parameter
	"""
	model_input_names = ["flattened_patches"]

	def __init__(
		self,
		do_convert_rgb: bool = True,
		do_normalize: bool = True,
		patch_size: Optional[Dict[str, int]] = None,
		max_total_patches: int = 2048,
		is_vqa: bool = True,
		**kwargs,
	) -> None:
		self.do_convert_rgb = do_convert_rgb
		self.do_normalize = do_normalize
		self.patch_size = patch_size if patch_size is not None else {"height": 16, "width": 16}
		self.max_total_patches = max_total_patches
		self.is_vqa = is_vqa

	def extract_flattened_patches(self, images: List[np.ndarray], input_data_format: Optional[Union[str, str]] = None, **kwargs) -> np.ndarray:
		"""
		Extract flattened patches from a list of images.
		"""
		flattened = extract_multi_image_flattened_patches(
			images=images,
			max_total_patches=self.max_total_patches,
			patch_size=self.patch_size,
			input_data_format=input_data_format,
		)
		return flattened

	def normalize(
		self,
		image: np.ndarray,
		data_format: Optional[Union[str, ChannelDimension]] = None,
		input_data_format: Optional[Union[str, ChannelDimension]] = None,
		**kwargs,
	) -> np.ndarray:
		if image.dtype == np.uint8:
			image = image.astype(np.float32)

		# take mean across the whole `image`
		mean = np.mean(image)
		std = np.std(image)
		adjusted_stddev = max(std, 1.0 / math.sqrt(np.prod(image.shape)))

		return normalize(
			image,
			mean=mean,
			std=adjusted_stddev,
			data_format=data_format,
			input_data_format=input_data_format,
			**kwargs,
		)

	def preprocess(
			self,
			images: Union[np.ndarray, List[np.ndarray]],
			header_text: Optional[str] = None,
			input_data_format: Optional[Union[str, str]] = None,
			**kwargs
		):
		"""
		Preprocess images and return a dictionary similar to BatchFeature.
		"""
		images = make_list_of_images(images)
		# if isinstance(images[0], Image.Image):
		# 	images = [np.array(image) for image in images]
		if not valid_images(images):
			raise ValueError("Invalid image type provided.")
		if self.do_convert_rgb:
			images = [convert_to_rgb(image) for image in images]

		images = [to_numpy_array(image) for image in images]
		if input_data_format is None:
			# We assume that all images have the same channel dimension format.
			input_data_format = infer_channel_dimension_format(images[0])

		if self.is_vqa:
			if header_text is None:
				raise ValueError("A header text must be provided for VQA models.")
			font_bytes = kwargs.pop("font_bytes", None)
			font_path = kwargs.pop("font_path", None)

			images[0] = render_header(images[0], header_text, font_bytes=font_bytes, font_path=font_path)

		if self.do_normalize:
			images = [self.normalize(image=image, input_data_format=input_data_format) for image in images]

		flattened_patches = self.extract_flattened_patches(images, input_data_format=input_data_format, **kwargs)
		attention_mask = (flattened_patches.sum(axis=-1) != 0).astype(np.float32)
		if kwargs.get("return_tensors", None) == "pt":
			flattened_patches = torch.from_numpy(flattened_patches).unsqueeze(0)
			attention_mask = torch.from_numpy(attention_mask).unsqueeze(0)
		return {"flattened_patches": flattened_patches, "attention_mask": attention_mask}

# ----------------------------
# Custom Processor (Image + Tokenizer)
# ----------------------------

class CustomPix2StructProcessor:
	"""
	A minimal processor that wraps a custom image processor and a tokenizer.
	"""
	def __init__(self, image_processor: CustomPix2StructImageProcessor, tokenizer: T5TokenizerFast) -> None:
		self.image_processor = image_processor
		self.tokenizer = tokenizer
		# Set tokenizer default: no token_type_ids.
		self.tokenizer.model_max_length = 512

	def __call__(
		self,
		images: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
		text: Optional[Union[str, List[str]]] = None,
		**kwargs
	):
		"""
		Process images with the custom image processor and optionally process text.
		Returns a dictionary that merges image and text encodings.
		"""
		if images is None and text is None:
			raise ValueError("Provide at least images or text.")

		out = {}
		if images is not None:
			# Process images.
			image_features = self.image_processor.preprocess(images, header_text=text, **kwargs)
			out.update(image_features)
		# if text is not None:
		# 	text_features = self.tokenizer(text=text, **kwargs)
		# 	# Rename keys to avoid clashing with the image processor's keys.
		# 	if "attention_mask" in text_features:
		# 		text_features["decoder_attention_mask"] = text_features.pop("attention_mask")
		# 	if "input_ids" in text_features:
		# 		text_features["decoder_input_ids"] = text_features.pop("input_ids")
		# 	out.update(text_features)
		return out

	def batch_decode(self, *args, **kwargs):
		return self.tokenizer.batch_decode(*args, **kwargs)

	def decode(self, *args, **kwargs):
		return self.tokenizer.decode(*args, **kwargs)

	@property
	def model_input_names(self):
		# Combine input names from tokenizer and image processor.
		t_names = getattr(self.tokenizer, "model_input_names", [])
		i_names = getattr(self.image_processor, "model_input_names", [])
		return list(dict.fromkeys(t_names + i_names))