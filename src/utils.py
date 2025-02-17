import ast
import random
from PIL import Image
import os
import yaml
import json
import argparse
import numpy as np
import torch
import difflib
from typing import Literal, Tuple, List

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="MP-DocVQA framework")

	# Required
	parser.add_argument("-m", "--model", type=str, required=True, help="Path to yml file with model configuration.")
	parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to yml file with dataset configuration.")

	# Optional
	parser.add_argument("--eval-start", action="store_true", default=True, help="Whether to evaluate the model before training or not.")
	parser.add_argument("--no-eval-start", dest="eval_start", action="store_false")

	# Overwrite config parameters
	parser.add_argument("-p", "--page-retrieval", type=str, help="Page retrieval set-up.")
	parser.add_argument("-bs", "--batch-size", type=int, help="DataLoader batch size.")
	parser.add_argument("-msl", "--max-sequence-length", type=int, help="Max input sequence length of the model.")
	parser.add_argument("--seed", type=int, help="Seed to allow reproducibility.")
	parser.add_argument("--save-dir", type=str, help="Seed to allow reproducibility.")

	parser.add_argument("--data-parallel", action="store_true", help="Boolean to overwrite data-parallel arg in config parallelize the execution.")
	parser.add_argument("--no-data-parallel", action="store_false", dest="data_parallel", help="Boolean to overwrite data-parallel arg in config to indicate to parallelize the execution.")
	return parser.parse_args()

def parse_multitype2list_arg(argument: str) -> list:
	if argument is None:
		return argument

	if "-" in argument and "[" in argument and "]" in argument:
		first, last = argument.strip("[]").split("-")
		argument = list(range(int(first), int(last)))
		return argument

	argument = ast.literal_eval(argument)

	if isinstance(argument, int):
		argument = [argument]

	elif isinstance(argument, list):
		argument = argument

	return argument

def save_json(path: str, data: dict, **kwargs):
	smart = kwargs.get("smart", False)
	smart_start_level = kwargs.get("smart_start_level", 1)

	if not os.path.exists(os.path.dirname(path)):
		os.makedirs(os.path.dirname(path))
	with open(path, "w+") as f:
		if smart:
			f.write(smart_json_dumps(data, indent=4, smart_start_level=smart_start_level, max_inline_length=1000))
		else:
			json.dump(data, f, indent=4)

def save_yaml(path: str, data: dict):
	if not os.path.exists(os.path.dirname(path)):
		os.makedirs(os.path.dirname(path))
	with open(path, "w+") as f:
		yaml.dump(data, f)

def seed_everything(seed: int):
	random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True

def check_config(config: dict) -> bool:
	model_name = config["model_name"].lower()

	if "page_retrieval" not in config:
		config["page_retrieval"] = "none"

	page_retrieval = config["page_retrieval"].lower()
	if model_name not in ["hi-layoutlmv3", "hi-lt5", "hi-vt5"] and page_retrieval == "custom":
		raise ValueError('"Custom" retrieval is not allowed for {:}'.format(model_name))

	elif model_name in ["hi-layoutlmv3, hilt5", "hi-lt5", "hivt5", "hi-vt5"] and page_retrieval in ["concat", "logits", "maxconf", "anyconf", "maxconfpage", "anyconfpage", "majorpage", "weightmajorpage", "anyconforacle"]:
		raise ValueError('Hierarchical model {:} can"t run on {:} retrieval type. Only "oracle" and "custom" are allowed.'.format(model_name, page_retrieval))

	if page_retrieval == "custom" and model_name not in ["hi-layoutlmv3", "hi-lt5", "hi-vt5"]:
		raise ValueError('"Custom" page retrieval only allowed for Heirarchical methods ("hi-layoutlmv3", "hi-lt5", "hi-vt5").')

	elif page_retrieval in ["concat", "logits", "maxconf", "anyconf", "maxconfpage", "anyconfpage", "majorpage", "weightmajorpage", "anyconforacle"] and config.get("max_pages") is not None:
		print("WARNING - Max pages ({:}) value is ignored for {:} page-retrieval setting.".format(config.get("max_pages"), page_retrieval))

	elif page_retrieval == "none" and config["dataset_name"] not in ["SP-DocVQA"]:
		print('Page retrieval can"t be none for dataset "{:s}". This is intended only for single page datasets. Please specify in the method config file the "page_retrieval" setup to one of the following: [oracle, concat, logits, maxconf, anyconf, maxconfpage, anyconfpage, majorpage, weightmajorpage, anyconforacle, custom] '.format(config["dataset_name"]))

	if "save_dir" in config:
		if not config["save_dir"].endswith("/"):
			config["save_dir"] = config["save_dir"] + "/"

		if not os.path.exists(config["save_dir"]):
			os.makedirs(config["save_dir"])

	return True

def load_config(args: argparse.Namespace) -> dict:
	if args.model == "HiVT5":
		args.embed_model = None
		args.chunk_num = None
		args.chunk_size = None
		args.overlap = None
		args.include_surroundings = None
	model_config_path = "configs/{:}.yml".format(args.model)
	dataset_config_path = "configs/{:}.yml".format(args.dataset)
	model_config = parse_config(yaml.safe_load(open(model_config_path, "r")), args)
	dataset_config = parse_config(yaml.safe_load(open(dataset_config_path, "r")), args)
	training_config = model_config.pop("training_parameters")

	# Append and overwrite config values from arguments.
	# config = {"dataset_params": dataset_config, "model_params": model_config, "training_params": training_config}
	config = {**dataset_config, **model_config, **training_config}

	config.update({k: v for k, v in args._get_kwargs() if v is not None})
	config.pop("model")
	config.pop("dataset")

	# Set default seed
	if "seed" not in config:
		print('Seed not specified. Setting default seed to "{:d}"'.format(42))
		config["seed"] = 42

	if "page_retrieval" not in config:
		config["page_retrieval"] = "concat"
	check_config(config)

	return config

def parse_config(config: dict, args: argparse.Namespace) -> dict:
	# Import included configs.
	for included_config_path in config.get("includes", []):
		config = load_config(included_config_path, args) | config
	return config

def correct_alignment(context: str, answer: str, start_idx: int, end_idx: int) -> Tuple[int, int]:

	if context[start_idx: end_idx] == answer:
		return [start_idx, end_idx]

	elif context[start_idx - 1: end_idx] == answer:
		return [start_idx - 1, end_idx]

	elif context[start_idx: end_idx + 1] == answer:
		return [start_idx, end_idx + 1]

	else:
		print(context[start_idx: end_idx], answer)
		return None

def time_stamp_to_hhmmss(timestamp: float, string: bool=True) -> str:
	hh = int(timestamp/3600)
	mm = int((timestamp-hh*3600)/60)
	ss = int(timestamp - hh*3600 - mm*60)

	time = "{:02d}:{:02d}:{:02d}".format(hh, mm, ss) if string else [hh, mm, ss]

	return time

def compute_grid(image_patches: List[Image.Image]) -> Tuple[int, int]:
	# Sort image patches by height (or width) to make strip packing more efficient
	image_patches = sorted(image_patches, key=lambda img: img.height, reverse=True)
	total_area = sum(im.width * im.height for im in image_patches)
	# Estimate optimal grid dimensions based on total area and aspect ratios
	grid_width = max(im.width for im in image_patches)
	grid_height = int(total_area / grid_width)
	return grid_width, grid_height

def concatenate_patches(
		image_patches: List[Image.Image],
		mode: Literal["horizontal", "vertical", "grid"] = "grid"
) -> Image.Image:
	if not image_patches:
		# Return a blank image
		return Image.new("RGB", (5, 5))
	widths, heights = zip(*(i.size for i in image_patches))
	if mode == "horizontal":
		# Concatenate images horizontally
		total_width = sum(widths)
		max_height = max(heights)
		new_image = Image.new("RGB", (total_width, max_height))
		x_offset = 0
		for im in image_patches:
			new_image.paste(im, (x_offset, 0))
			x_offset += im.size[0]
	elif mode == "vertical":
		# Concatenate images vertically
		max_width = max(widths)
		total_height = sum(heights)
		new_image = Image.new("RGB", (max_width, total_height))
		y_offset = 0
		for im in image_patches:
			new_image.paste(im, (0, y_offset))
			y_offset += im.size[1]
	elif mode == "grid":
		# Concatenate images in a compact grid layout
		grid_width, grid_height = compute_grid(image_patches)
		new_image = Image.new("RGB", size=(grid_width, grid_height))
		x_offset, y_offset = 0, 0
		row_height = 0
		# Place the images in rows (strip-packing style)
		for img in image_patches:
			if x_offset + img.width > grid_width:
				# Move to the next row
				x_offset = 0
				y_offset += row_height
				row_height = 0
			new_image.paste(img, (x_offset, y_offset))
			x_offset += img.width
			row_height = max(row_height, img.height)
	return new_image

def flatten(
		lst: List[list],
		add_sep_token: bool = True
	) -> list:
	if add_sep_token:
		# Add a separator token between the sublists
		flat_list = []
		for i, sublist in enumerate(lst):
			if len(sublist) == 0:
				continue
			if i > 0:
				if isinstance(sublist[0], str): # for words
					flat_list.append("<sep>")
				elif isinstance(sublist[0], list): # for boxes
					flat_list.append([0, 0, 0, 0])
				elif isinstance(sublist[0], int): # for layout labels
					flat_list.append(0)
			flat_list.extend(sublist)
		return flat_list
	else:
		return [item for sublist in lst for item in sublist]

def get_similarity_score(a: str, b: str):
	"""
	Calculate the highest similarity score between string b and any substring of a.
	
	Parameters:
	a (str): The long text.
	b (str): The short string to match.
	
	Returns:
	float: A score between 0 and 1 representing the similarity.
	"""
	a = a.lower()
	b = b.lower()
	best_score = 0.0
	len_b = len(b)
	len_a = len(a)
	
	for i in range(len_a - len_b + 1):
		substring = a[i:i+len_b]
		# Compute similarity ratio
		score = difflib.SequenceMatcher(None, b, substring).ratio()
		if score > best_score:
			best_score = score
			if best_score == 1.0:
				# Exact match found, no need to continue
				break
	return np.log(best_score + 1) / np.log(2)

def compute_iou(box: List[float], boxes: np.ndarray) -> np.ndarray:
	"""
	Compute the IoU of a box with an array of boxes.
	Each box is in [xmin, ymin, xmax, ymax] format.
	"""
	xx1 = np.maximum(box[0], boxes[:, 0])
	yy1 = np.maximum(box[1], boxes[:, 1])
	xx2 = np.minimum(box[2], boxes[:, 2])
	yy2 = np.minimum(box[3], boxes[:, 3])
	w = np.maximum(0, xx2 - xx1)
	h = np.maximum(0, yy2 - yy1)
	inter = w * h
	area_box = (box[2] - box[0]) * (box[3] - box[1])
	areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
	iou = inter / (area_box + areas - inter + 1e-8)
	return iou

def non_maximum_suppression(
	boxes: List[List[float]], 
	iou_threshold: float = 0.7
) -> List[int]:
	"""
	Run NMS over all boxes; return indices of boxes to keep.
	Boxes should be in [xmin, ymin, xmax, ymax] format.
	"""
	if not boxes:
		return []
	
	boxes_arr = np.array(boxes)
	areas = (boxes_arr[:, 2] - boxes_arr[:, 0]) * (boxes_arr[:, 3] - boxes_arr[:, 1])
	order = areas.argsort()[::-1]
	keep = []
	
	while order.size > 0:
		idx = order[0]
		keep.append(idx)
		if order.size == 1:
			break
		remaining = order[1:]
		ious = compute_iou(boxes_arr[idx], boxes_arr[remaining])
		inds = np.where(ious <= iou_threshold)[0]
		order = order[inds + 1]
	
	return keep

def containment_ratio(
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

def is_simple(obj):
	"""
	Returns True if the object is simple enough to be dumped in a single line.
	'Simple' means:
	- It is a primitive (str, int, float, bool, None)
	- It is a list whose items are all simple
	- It is a dict whose keys (assumed to be strings) and values are all simple
	"""
	if isinstance(obj, (str, int, float, bool)) or obj is None:
		return True
	elif isinstance(obj, list):
		return all(is_simple(item) for item in obj)
	elif isinstance(obj, dict):
		return all(isinstance(k, str) and is_simple(v) for k, v in obj.items())
	return False

def smart_json_dumps(obj, indent=2, level=0, max_inline_length=80, smart_start_level=0):
	"""
	Recursively dumps a JSON object with smart formatting.
	
	The parameter `smart_start_level` defines the level from which smart formatting is applied:
	- If the current recursion level is less than `smart_start_level`, the object is formatted in
		a normal (always expanded) multi-line way.
	- If the current level is >= smart_start_level, then:
			- If the object is "simple" and its inline JSON representation is not too long,
			it is printed inline.
			- Otherwise, it is formatted with newlines and proper indentation.
	
	:param obj: The Python object to dump.
	:param indent: Number of spaces to use for each indentation level.
	:param level: The current recursion level (used internally).
	:param max_inline_length: Maximum length of inline JSON string for it to be kept on one line.
	:param smart_start_level: The recursion level at which smart inline formatting begins.
	"""
	current_indent = ' ' * (indent * level)
	next_indent = ' ' * (indent * (level + 1))
	
	# When we haven't reached the smart level, format normally (always expanded)
	if level < smart_start_level:
		if isinstance(obj, dict):
			if not obj:
				return '{}'
			items = []
			for k, v in obj.items():
				if isinstance(k, int) or isinstance(k, float):
					key_str = json.dumps(str(k), ensure_ascii=False)
				else:
					key_str = json.dumps(k, ensure_ascii=False)

				# Recursively format children; they might be smart if level+1 >= smart_start_level.
				value_str = smart_json_dumps(v, indent, level + 1, max_inline_length, smart_start_level)
				items.append(f"{next_indent}{key_str}: {value_str}")
			return "{\n" + ",\n".join(items) + "\n" + current_indent + "}"
		
		elif isinstance(obj, list):
			if not obj:
				return '[]'
			items = []
			for item in obj:
				item_str = smart_json_dumps(item, indent, level + 1, max_inline_length, smart_start_level)
				items.append(f"{next_indent}{item_str}")
			return "[\n" + ",\n".join(items) + "\n" + current_indent + "]"
		
		else:
			return json.dumps(obj, ensure_ascii=False)
	
	# At levels where smart formatting applies:
	if is_simple(obj):
		inline_repr = json.dumps(obj, ensure_ascii=False)
		if len(inline_repr) <= max_inline_length:
			return inline_repr

	if isinstance(obj, dict):
		if not obj:
			return '{}'
		items = []
		for k, v in obj.items():
			if isinstance(k, int):
				key_str = json.dumps(str(k), ensure_ascii=False)
			else:
				key_str = json.dumps(k, ensure_ascii=False)

			value_str = smart_json_dumps(v, indent, level + 1, max_inline_length, smart_start_level)
			items.append(f"{next_indent}{key_str}: {value_str}")
		return "{\n" + ",\n".join(items) + "\n" + current_indent + "}"
	
	elif isinstance(obj, list):
		if not obj:
			return '[]'
		items = []
		for item in obj:
			item_str = smart_json_dumps(item, indent, level + 1, max_inline_length, smart_start_level)
			items.append(f"{next_indent}{item_str}")
		return "[\n" + ",\n".join(items) + "\n" + current_indent + "]"
	
	# For any other types, fall back to json.dumps.
	return json.dumps(obj, ensure_ascii=False)
