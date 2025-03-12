import os
import random
import json
from PIL import Image
from typing import Any, List, Tuple, Dict
from torch.utils.data import Dataset
from time import time

class Infographics(Dataset):

	def __init__(
			self,
			config: dict,
	):
		json_dir = config["json_dir"]
		images_dir = config["images_dir"]
		ocr_dir = config["ocr_dir"]
		split = config["split"]
		if split == "val":
			qas_file = os.path.join(json_dir, "infographicsVQA_val_v1.0_withQT.json")
		else:
			qas_file = os.path.join(json_dir, f"infographicsVQA_{split}_v1.0.json")
		with open(qas_file, "r") as f:
			self.qas_data = json.load(f)["data"]
		
		self.page_retrieval = config["page_retrieval"].lower()
		assert(self.page_retrieval in ["oracle", "concat", "logits", "custom", "maxconf", "anyconf", "maxconfpage", "anyconfpage", "majorpage", "weightmajorpage", "anyconforacle"])

		self.images_dir = images_dir
		self.ocr_dir = ocr_dir

		self.use_images = config.get("use_images", False)
		self.get_raw_ocr_data = config.get("get_raw_ocr_data", False)
		self.max_pages = config.get("max_pages", 1)

	def __len__(self):
		return len(self.qas_data)
	
	def sample(
			self,
			idx: int = None,
			question_id: int = None
	) -> Dict[str, Any]:

		if idx is not None:
			return self.__getitem__(idx)

		if question_id is not None:
			for idx in range(self.__len__()):
				record = self.qas_data[idx]
				if record["questionId"] == question_id:
					return self.__getitem__(idx)
				
			raise ValueError(f"Question ID {question_id} not found in the dataset.")

		idx = random.randint(0, self.__len__())
		return self.__getitem__(idx)
	
	def __getitem__(self, idx: int) -> Dict[str, Any]:
		start_time = time()
		record = self.qas_data[idx]

		question = record["question"]
		answers = list(set(answer.lower() for answer in record.get("answers", [""])))
		answer_page_idx = 0
		num_pages = 1

		ocr_file = os.path.join(self.ocr_dir, record["ocr_output_file"])
		with open(ocr_file, "r") as f:
			ocr_data = json.load(f)
		if "LINE" in ocr_data:
			context = [" ".join([ocr["Text"].lower() for ocr in ocr_data["LINE"]])]
		else:
			context = []

		if self.use_images:
			image_names = [os.path.join(self.images_dir, record["image_local_name"])]
			images = [Image.open(image_names[0]).convert("RGB")]

		def get_box(polygon):
			box = [
				polygon[0]["X"],
				polygon[0]["Y"],
				polygon[2]["X"],
				polygon[2]["Y"]
			]
			return box

		if self.get_raw_ocr_data:
			if "WORD" in ocr_data:
				words = [[ocr["Text"].lower() for ocr in ocr_data["WORD"]]]
				boxes = [[get_box(ocr["Geometry"]["Polygon"]) for ocr in ocr_data["WORD"]]]
			else:
				words = [[]]
				boxes = [[]]

		if context:
			start_idxs, end_idxs = self._get_start_end_idx(context[answer_page_idx], answers)
		else:
			start_idxs, end_idxs = 0, 0

		sample_info = {
			"question_id": record["questionId"],
			"questions": question,
			"contexts": context,
			"context_page_corresp": None,
			"answers": answers,
			"answer_page_idx": answer_page_idx,
			"num_pages": num_pages,
			"load_time": time()-start_time
		}

		if self.use_images:
			sample_info["image_names"] = image_names
			sample_info["images"] = images

		if self.get_raw_ocr_data:
			sample_info["words"] = words
			sample_info["boxes"] = boxes
		else:  # Information for extractive models
			sample_info["start_indxs"] = start_idxs
			sample_info["end_indxs"] = end_idxs

		return sample_info

	def _get_start_end_idx(
			self,
			context: str,
			answers: List[str]
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

def infographics_collate_fn(batch: List[dict]) -> dict:  # It"s actually the same as in SP-DocVQA...
	batch = {k: [dic[k] for dic in batch] for k in batch[0]}  # List of dictionaries to dict of lists.
	return batch
