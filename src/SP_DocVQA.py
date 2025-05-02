import os
import random
from PIL import Image
import numpy as np
from typing import Literal, Any, Dict, List, Tuple
from torch.utils.data import Dataset

class SPDocVQA(Dataset):

	def __init__(
			self,
			config: dict,
	):
		imdb_dir = config["imdb_dir"]
		images_dir = config["images_dir"]
		split = config["split"]
		data = np.load(os.path.join(imdb_dir, "imdb_{:s}.npy".format(split)), allow_pickle=True)
		self.header = data[0]
		self.imdb = data[1:]
		self.hierarchical_method = config.get("hierarchical_method", True)

		self.max_answers = 2
		self.images_dir = images_dir

		self.use_images = config.get("use_images", False)
		self.get_raw_ocr_data = config.get("get_raw_ocr_data", False)

		self.use_precomputed_layouts = config.get("use_precomputed_layouts", False)
		if self.use_precomputed_layouts:
			layouts_file = config["precomputed_layouts_path"]
			self.layout_info = np.load(layouts_file, allow_pickle=True)

	def __len__(self):
		return len(self.imdb)

	def __getitem__(self, idx: int) ->Dict[str, Any]:
		record = self.imdb[idx]
		question = record["question"]
		context = " ".join([word.lower() for word in record["ocr_tokens"]])
		context_page_corresp = [0 for ix in range(len(context))]  # This is used to predict the answer page in MP-DocVQA. To keep it simple, use a mock list with corresponding page to 0.

		answers = list(set(answer.lower() for answer in record["answers"]))

		if self.use_images:
			image_name = os.path.join(self.images_dir, "{:s}.png".format(record["image_name"]))
			image = Image.open(image_name).convert("RGB")
			if self.use_precomputed_layouts:
				img_name = os.path.splitext(record["image_name"])[0]
				layouts = [self.layout_info[img_name].item()]

		if self.get_raw_ocr_data:
			words = [word.lower() for word in record["ocr_tokens"]]
			boxes = np.array([bbox for bbox in record["ocr_normalized_boxes"]])

		if self.hierarchical_method:
			words = [words]
			boxes = [boxes]
			image_name = [image_name]
			image = [image]

		start_idxs, end_idxs = self._get_start_end_idx(context, answers)

		sample_info = {"question_id": record["question_id"],
					   "questions": question,
					   "contexts": context,
					   "answers": answers,
					   "start_indxs": start_idxs,
					   "end_indxs": end_idxs
					   }

		if self.use_images:
			sample_info["image_names"] = image_name
			sample_info["images"] = image
			if self.use_precomputed_layouts:
				sample_info["layouts"] = layouts

		if self.get_raw_ocr_data:
			sample_info["words"] = words
			sample_info["boxes"] = boxes
			sample_info["num_pages"] = 1
			sample_info["answer_page_idx"] = 0

		else:  # Information for extractive models
			sample_info["context_page_corresp"] = context_page_corresp
			sample_info["start_indxs"] = start_idxs
			sample_info["end_indxs"] = end_idxs

		return sample_info

	def _get_start_end_idx(
			self,
			context: str,
			answers:List[str]
	) -> Tuple[int, int]:
		answer_positions = []
		for answer in answers:
			start_idx = context.find(answer)

			if start_idx != -1:
				end_idx = start_idx + len(answer)
				answer_positions.append([start_idx, end_idx])

		if len(answer_positions) > 0:
			start_idx, end_idx = random.choice(answer_positions)  # If both answers are in the context. Choose one randomly.
		else:
			start_idx, end_idx = 0, 0  # If the indices are out of the sequence length they are ignored. Therefore, we set them as a very big number.

		return start_idx, end_idx

def singlepage_docvqa_collate_fn(batch:List[dict]) -> dict:
	batch = {k: [dic[k] for dic in batch] for k in batch[0]}  # List of dictionaries to dict of lists.
	return batch
