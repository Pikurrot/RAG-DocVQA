import os
import random
from PIL import Image
import numpy as np
from typing import Any, List, Tuple, Dict
from torch.utils.data import Dataset
from time import time
import random
from copy import deepcopy

class MPDocVQA(Dataset):

	def __init__(
			self,
			config: dict,
	):
		imdb_dir = config["imdb_dir"]
		images_dir = config["images_dir"]
		page_retrieval = config["page_retrieval"]
		split = config["split"]
		data = np.load(os.path.join(imdb_dir, "imdb_{:s}.npy".format(split)), allow_pickle=True)
		self.header = data[0]
		self.imdb = data[1:]
		size = config.get("size", 1.0)
		if isinstance(size, float) and size < 1.0:
			self.imdb = self.imdb[:int(size*len(self.imdb))]
		elif isinstance(size, tuple):
			self.imdb = self.imdb[int(size[0]*len(self.imdb)):int(size[1]*len(self.imdb))]

		self.page_retrieval = page_retrieval.lower()
		assert(self.page_retrieval in ["oracle", "concat", "logits", "custom", "maxconf", "anyconf", "maxconfpage", "anyconfpage", "majorpage", "weightmajorpage", "anyconforacle"])

		self.max_answers = 2
		self.images_dir = images_dir

		self.use_images = config.get("use_images", False)
		self.get_raw_ocr_data = config.get("get_raw_ocr_data", False)
		self.max_pages = config.get("max_pages", 1)

		self.use_precomputed_layouts = config.get("use_precomputed_layouts", False)
		if self.use_precomputed_layouts:
			layouts_file = config["precomputed_layouts_path"]
			self.layout_info = np.load(layouts_file, allow_pickle=True)

	def __len__(self):
		return len(self.imdb)

	def sample(
			self,
			idx: int = None,
			question_id: int = None
	) -> Dict[str, Any]:

		if idx is not None:
			return self.__getitem__(idx)

		if question_id is not None:
			for idx in range(self.__len__()):
				record = self.imdb[idx]
				if record["question_id"] == question_id:
					return self.__getitem__(idx)

			raise ValueError("Question ID {:d} not in dataset.".format(question_id))

		idx = random.randint(0, self.__len__())
		return self.__getitem__(idx)

	def __getitem__(self, idx: int) -> Dict[str, Any]:
		start_time = time()
		record = self.imdb[idx]

		question = record["question"]
		answers = list(set(answer.lower() for answer in record.get("answers", [""])))
		answer_page_idx = record.get("answer_page_idx", 0)
		num_pages = record["imdb_doc_pages"]

		if self.page_retrieval in ["oracle", "anyconforacle"]:
			context = [" ".join([word.lower() for word in record["ocr_tokens"][answer_page_idx]])]
			context_page_corresp = None
			num_pages = 1

			if self.use_images:
				image_names = os.path.join(self.images_dir, "{:s}.jpg".format(record['image_name'][answer_page_idx]))
				images = [Image.open(image_names).convert("RGB")]
				if self.use_precomputed_layouts:
					layouts = [self.layout_info[record['image_name'][answer_page_idx]].item()]

			if self.get_raw_ocr_data:
				words = [[word.lower() for word in record['ocr_tokens'][answer_page_idx]]]
				boxes = [record['ocr_normalized_boxes'][answer_page_idx]]
			
			start_idxs, end_idxs = self._get_start_end_idx(context[0], answers)
		
		elif self.page_retrieval in ["concat", "logits", "maxconf", "anyconf", "maxconfpage", "anyconfpage", "majorpage", "weightmajorpage"]:
			context = []
			for page_ix in range(record["imdb_doc_pages"]):
				context.append(" ".join([word.lower() for word in record["ocr_tokens"][page_ix]]))

			context_page_corresp = None

			if self.use_images:
				image_names = [os.path.join(self.images_dir, "{:s}.jpg".format(image_name)) for image_name in record["image_name"]]
				images = [Image.open(img_path).convert("RGB") for img_path in image_names]
				if self.use_precomputed_layouts:
					layouts = [self.layout_info[image_name].item() for image_name in record["image_name"]]

			if self.get_raw_ocr_data:
				words = []
				boxes = record["ocr_normalized_boxes"]
				for p in range(num_pages):
					words.append([word.lower() for word in record["ocr_tokens"][p]])
			
			start_idxs, end_idxs = self._get_start_end_idx(context[answer_page_idx], answers)

		elif self.page_retrieval == "custom":
			first_page, last_page = self.get_pages(record)
			answer_page_idx = answer_page_idx - first_page
			num_pages = len(range(first_page, last_page))

			words = []
			boxes = []
			context = []
			image_names = []

			for page_ix in range(first_page, last_page):
				words.append([word.lower() for word in record['ocr_tokens'][page_ix]])
				boxes.append(np.array(record['ocr_normalized_boxes'][page_ix], dtype=np.float32))
				context.append(' '.join([word.lower() for word in record['ocr_tokens'][page_ix]]))
				image_names.append(os.path.join(self.images_dir, "{:s}.jpg".format(record['image_name'][page_ix])))

			context_page_corresp = None

			if num_pages < self.max_pages:
				for _ in range(self.max_pages - num_pages):
					words.append([''])
					boxes.append(np.zeros([1, 4], dtype=np.float32))

			if self.use_images:
				images = [Image.open(img_path).convert("RGB") for img_path in image_names]
				images += [Image.new('RGB', (2, 2)) for i in range(self.max_pages - len(image_names))]  # Pad with 2x2 images.
				if self.use_precomputed_layouts:
					layouts = [self.layout_info[image_name].item() for image_name in record["image_name"]]
					layouts += [None for i in range(self.max_pages - len(layouts))]  # Pad with None layouts.
				
			start_idxs, end_idxs = None, None


		sample_info = {
			"question_id": record["question_id"],
			"questions": question,
			"contexts": context,
			"context_page_corresp": context_page_corresp,
			"answers": answers,
			"answer_page_idx": answer_page_idx,
			"num_pages": num_pages,
			"load_time": time()-start_time
		}

		if self.use_images:
			sample_info["image_names"] = image_names
			sample_info["images"] = images
			if self.use_precomputed_layouts:
				sample_info["layouts"] = layouts

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

	def get_pages(self, sample_info: dict) -> Tuple[int, int]:
		# TODO implement margins
		answer_page = sample_info.get("answer_page_idx", 0)
		document_pages = sample_info["imdb_doc_pages"]
		if document_pages <= self.max_pages:
			first_page, last_page = 0, document_pages

		else:
			first_page_lower_bound = max(0, answer_page-self.max_pages+1)
			first_page_upper_bound = answer_page
			first_page = random.randint(first_page_lower_bound, first_page_upper_bound)
			last_page = first_page + self.max_pages

			if last_page > document_pages:
				last_page = document_pages
				first_page = last_page-self.max_pages

			try:
				assert (answer_page in range(first_page, last_page))  # answer page is in selected range.
				assert (last_page-first_page == self.max_pages)  # length of selected range is correct.
			except AssertionError:
				assert (answer_page in range(first_page, last_page))  # answer page is in selected range.
				assert (last_page - first_page == self.max_pages)  # length of selected range is correct.
		assert(answer_page in range(first_page, last_page))
		assert(first_page >= 0)
		assert(last_page <= document_pages)

		return first_page, last_page


class MPDocVQA_NoisePages(Dataset):

	def __init__(
			self,
			config: dict,
	):
		imdb_dir = config["imdb_dir"]
		images_dir = config["images_dir"]
		page_retrieval = config["page_retrieval"]
		split = config["split"]
		data = np.load(os.path.join(imdb_dir, "imdb_{:s}.npy".format(split)), allow_pickle=True)
		self.header = data[0]
		self.imdb = data[1:]
		size = config.get("size", 1.0)
		if isinstance(size, float) and size < 1.0:
			self.imdb = self.imdb[:int(size*len(self.imdb))]
		elif isinstance(size, tuple):
			self.imdb = self.imdb[int(size[0]*len(self.imdb)):int(size[1]*len(self.imdb))]
		elif isinstance(size, int) and size > 1:
			# Proportional sampling by number of pages
			self.imdb = self._proportional_sampling_by_pages(self.imdb, size)

		self.page_retrieval = page_retrieval.lower()
		assert(self.page_retrieval in ["oracle", "concat", "logits", "custom", "maxconf", "anyconf", "maxconfpage", "anyconfpage", "majorpage", "weightmajorpage", "anyconforacle"])

		self.max_answers = 2
		self.images_dir = images_dir

		self.use_images = config.get("use_images", False)
		self.get_raw_ocr_data = config.get("get_raw_ocr_data", False)
		self.max_pages = config.get("max_pages", 1)
		self.noise_pages = config.get("noise_pages", 0)
		self.noise_seed = config.get("noise_seed", 42)
		self.rng = random.Random(self.noise_seed)

		self.use_precomputed_layouts = config.get("use_precomputed_layouts", False)
		if self.use_precomputed_layouts:
			layouts_file = config["precomputed_layouts_path"]
			self.layout_info = np.load(layouts_file, allow_pickle=True)

		if self.noise_pages > 0 and self.page_retrieval == "concat":
			def doc_id(rec):
				return rec['image_id']

			seed_doc_ids = {doc_id(rec) for rec in self.imdb}
			# pool = list of (ocr_tokens, image_name, boxes, ...)
			self.noise_pool = [
				deepcopy(rec) for rec in data[1:]   # full dataset
				if doc_id(rec) not in seed_doc_ids
			]
			assert len(self.noise_pool) >= self.noise_pages, \
				f"Noise pool too small for requested N={self.noise_pages}. Only {len(self.noise_pool)} pages available."

	def _proportional_sampling_by_pages(self, imdb_data, target_size):
		"""
		Sample data proportionally by number of pages to ensure equal representation.
		"""
		# Group samples by number of pages
		samples_by_pages = {}
		for sample in imdb_data:
			num_pages = len(sample["image_name"])
			if num_pages not in samples_by_pages:
				samples_by_pages[num_pages] = []
			samples_by_pages[num_pages].append(sample)
		
		# Calculate target samples per page count
		num_page_groups = len(samples_by_pages)
		samples_per_group = target_size // num_page_groups
		remaining_samples = target_size % num_page_groups
		
		# First pass: determine how many samples each group can actually provide
		actual_samples_per_group = {}
		shortfall = 0
		
		for i, (pages, samples) in enumerate(sorted(samples_by_pages.items())):
			# Add one extra sample to some groups to distribute remaining samples
			target_for_group = samples_per_group + (1 if i < remaining_samples else 0)
			
			if len(samples) >= target_for_group:
				actual_samples_per_group[pages] = target_for_group
			else:
				# This group has fewer samples than needed
				actual_samples_per_group[pages] = len(samples)
				shortfall += target_for_group - len(samples)
		
		# Second pass: redistribute the shortfall to groups that can provide more
		if shortfall > 0:
			# Find groups that can provide additional samples
			groups_with_extra = []
			for pages, samples in samples_by_pages.items():
				available_extra = len(samples) - actual_samples_per_group[pages]
				if available_extra > 0:
					groups_with_extra.append(pages)
			
			# Redistribute shortfall using round-robin
			remaining_shortfall = shortfall
			round_robin_index = 0
			
			while remaining_shortfall > 0 and groups_with_extra:
				# Get current group in round-robin
				current_pages = groups_with_extra[round_robin_index]
				
				# Check if this group still has capacity
				available_extra = len(samples_by_pages[current_pages]) - actual_samples_per_group[current_pages]
				
				if available_extra > 0:
					# Assign one sample to this group
					actual_samples_per_group[current_pages] += 1
					remaining_shortfall -= 1
				else:
					# This group is now full, remove it from the list
					groups_with_extra.remove(current_pages)
					# Adjust index if we removed an element before current position
					if round_robin_index >= len(groups_with_extra) and groups_with_extra:
						round_robin_index = 0
					continue
				
				# Move to next group in round-robin
				round_robin_index = (round_robin_index + 1) % len(groups_with_extra)
		
		# Sample from each group according to the final allocation
		sampled_data = []
		rng = random.Random(42)  # Fixed seed for reproducible sampling
		
		for pages, target_count in actual_samples_per_group.items():
			if target_count > 0:
				group_samples = rng.sample(samples_by_pages[pages], target_count)
				sampled_data.extend(group_samples)
		
		# Print final distribution
		final_distribution = {}
		for sample in sampled_data:
			num_pages = len(sample["image_name"])
			final_distribution[num_pages] = final_distribution.get(num_pages, 0) + 1
		
		print(f"Final distribution after sampling {len(sampled_data)} samples:")
		for pages in sorted(final_distribution.keys()):
			print(f"  {pages} pages: {final_distribution[pages]} samples")
		
		return sampled_data

	def __len__(self):
		return len(self.imdb)

	def sample(
			self,
			idx: int = None,
			question_id: int = None
	) -> Dict[str, Any]:

		if idx is not None:
			return self.__getitem__(idx)

		if question_id is not None:
			for idx in range(self.__len__()):
				record = self.imdb[idx]
				if record["question_id"] == question_id:
					return self.__getitem__(idx)

			raise ValueError("Question ID {:d} not in dataset.".format(question_id))

		idx = random.randint(0, self.__len__())
		return self.__getitem__(idx)

	def __getitem__(self, idx: int) -> Dict[str, Any]:
		start_time = time()
		record = deepcopy(self.imdb[idx])

		question = record["question"]
		answers = list(set(answer.lower() for answer in record.get("answers", [""])))
		answer_page_idx = record.get("answer_page_idx", 0)
		num_pages = record["imdb_doc_pages"]

		if self.page_retrieval in ["oracle", "anyconforacle"]:
			context = [" ".join([word.lower() for word in record["ocr_tokens"][answer_page_idx]])]
			context_page_corresp = None
			num_pages = 1

			if self.use_images:
				image_names = os.path.join(self.images_dir, "{:s}.jpg".format(record['image_name'][answer_page_idx]))
				images = [Image.open(image_names).convert("RGB")]
				if self.use_precomputed_layouts:
					layouts = [self.layout_info[record['image_name'][answer_page_idx]].item()]

			if self.get_raw_ocr_data:
				words = [[word.lower() for word in record['ocr_tokens'][answer_page_idx]]]
				boxes = [record['ocr_normalized_boxes'][answer_page_idx]]
			
			start_idxs, end_idxs = self._get_start_end_idx(context[0], answers)
		
		elif self.page_retrieval in ["concat", "logits", "maxconf", "anyconf", "maxconfpage", "anyconfpage", "majorpage", "weightmajorpage"]:
			context = []
			for page_ix in range(record["imdb_doc_pages"]):
				context.append(" ".join([word.lower() for word in record["ocr_tokens"][page_ix]]))

			context_page_corresp = None

			if self.use_images:
				image_names = [os.path.join(self.images_dir, "{:s}.jpg".format(image_name)) for image_name in record["image_name"]]
				images = [Image.open(img_path).convert("RGB") for img_path in image_names]
				if self.use_precomputed_layouts:
					layouts = [self.layout_info[image_name].item() for image_name in record["image_name"]]

			if self.get_raw_ocr_data:
				words = []
				boxes = record["ocr_normalized_boxes"]
				for p in range(num_pages):
					words.append([word.lower() for word in record["ocr_tokens"][p]])
			
			start_idxs, end_idxs = self._get_start_end_idx(context[answer_page_idx], answers)

			# Add noise pages
			if self.noise_pages > 0:
				noise_records = self.rng.sample(self.noise_pool,
												self.noise_pages)
				for nrec in noise_records:
					# append OCR text
					context.append(" ".join(
						w.lower() for w in nrec["ocr_tokens"][0]))
					# append boxes / words / images if requested
					if self.get_raw_ocr_data:
						words.append([w.lower() for w in
									nrec["ocr_tokens"][0]])
						boxes.append(nrec["ocr_normalized_boxes"][0])
					if self.use_images:
						n_img_path = os.path.join(self.images_dir,
								f"{nrec['image_name'][0]}.jpg")
						images.append(Image.open(n_img_path).convert("RGB"))
						if self.use_precomputed_layouts:
							layouts.append(self.layout_info
										[nrec['image_name'][0]].item())

		elif self.page_retrieval == "custom":
			first_page, last_page = self.get_pages(record)
			answer_page_idx = answer_page_idx - first_page
			num_pages = len(range(first_page, last_page))

			words = []
			boxes = []
			context = []
			image_names = []

			for page_ix in range(first_page, last_page):
				words.append([word.lower() for word in record['ocr_tokens'][page_ix]])
				boxes.append(np.array(record['ocr_normalized_boxes'][page_ix], dtype=np.float32))
				context.append(' '.join([word.lower() for word in record['ocr_tokens'][page_ix]]))
				image_names.append(os.path.join(self.images_dir, "{:s}.jpg".format(record['image_name'][page_ix])))

			context_page_corresp = None

			if num_pages < self.max_pages:
				for _ in range(self.max_pages - num_pages):
					words.append([''])
					boxes.append(np.zeros([1, 4], dtype=np.float32))

			if self.use_images:
				images = [Image.open(img_path).convert("RGB") for img_path in image_names]
				images += [Image.new('RGB', (2, 2)) for i in range(self.max_pages - len(image_names))]  # Pad with 2x2 images.
				if self.use_precomputed_layouts:
					layouts = [self.layout_info[image_name].item() for image_name in record["image_name"]]
					layouts += [None for i in range(self.max_pages - len(layouts))]  # Pad with None layouts.
				
			start_idxs, end_idxs = None, None


		sample_info = {
			"question_id": record["question_id"],
			"questions": question,
			"contexts": context,
			"context_page_corresp": context_page_corresp,
			"answers": answers,
			"answer_page_idx": answer_page_idx,
			"num_pages": num_pages + self.noise_pages,
			"num_noise_pages": self.noise_pages,
			"num_seed_pages": num_pages,
			"load_time": time()-start_time
		}

		if self.use_images:
			sample_info["image_names"] = image_names
			sample_info["images"] = images
			if self.use_precomputed_layouts:
				sample_info["layouts"] = layouts

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

	def get_pages(self, sample_info: dict) -> Tuple[int, int]:
		# TODO implement margins
		answer_page = sample_info.get("answer_page_idx", 0)
		document_pages = sample_info["imdb_doc_pages"]
		if document_pages <= self.max_pages:
			first_page, last_page = 0, document_pages

		else:
			first_page_lower_bound = max(0, answer_page-self.max_pages+1)
			first_page_upper_bound = answer_page
			first_page = random.randint(first_page_lower_bound, first_page_upper_bound)
			last_page = first_page + self.max_pages

			if last_page > document_pages:
				last_page = document_pages
				first_page = last_page-self.max_pages

			try:
				assert (answer_page in range(first_page, last_page))  # answer page is in selected range.
				assert (last_page-first_page == self.max_pages)  # length of selected range is correct.
			except AssertionError:
				assert (answer_page in range(first_page, last_page))  # answer page is in selected range.
				assert (last_page - first_page == self.max_pages)  # length of selected range is correct.
		assert(answer_page in range(first_page, last_page))
		assert(first_page >= 0)
		assert(last_page <= document_pages)

		return first_page, last_page


def mpdocvqa_collate_fn(batch: List[dict]) -> dict:  # It"s actually the same as in SP-DocVQA...
	batch = {k: [dic[k] for dic in batch] for k in batch[0]}  # List of dictionaries to dict of lists.
	return batch
