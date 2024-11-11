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

def save_json(path: str, data: dict):
	if not os.path.exists(os.path.dirname(path)):
		os.makedirs(os.path.dirname(path))
	with open(path, "w+") as f:
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

	elif model_name in ["hi-layoutlmv3, hilt5", "hi-lt5", "hivt5", "hi-vt5"] and page_retrieval in ["concat", "logits", "maxconf", "anyconf"]:
		raise ValueError('Hierarchical model {:} can"t run on {:} retrieval type. Only "oracle" and "custom" are allowed.'.format(model_name, page_retrieval))

	if page_retrieval == "custom" and model_name not in ["hi-layoutlmv3", "hi-lt5", "hi-vt5"]:
		raise ValueError('"Custom" page retrieval only allowed for Heirarchical methods ("hi-layoutlmv3", "hi-lt5", "hi-vt5").')

	elif page_retrieval in ["concat", "logits", "maxconf", "anyconf"] and config.get("max_pages") is not None:
		print("WARNING - Max pages ({:}) value is ignored for {:} page-retrieval setting.".format(config.get("max_pages"), page_retrieval))

	elif page_retrieval == "none" and config["dataset_name"] not in ["SP-DocVQA"]:
		print('Page retrieval can"t be none for dataset "{:s}". This is intended only for single page datasets. Please specify in the method config file the "page_retrieval" setup to one of the following: [oracle, concat, logits, maxconf, anyconf, custom] '.format(config["dataset_name"]))

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
				else: # for boxes
					flat_list.append([0, 0, 0, 0])
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
