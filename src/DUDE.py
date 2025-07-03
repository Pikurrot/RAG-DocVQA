import os
import io
import random
from PIL import Image
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
from typing import Any, Dict, List
from time import time
from tqdm import tqdm

class DUDE(Dataset):

	def __init__(
			self,
			config: dict,
	):
		self.dataset = build_dude(config, config["split"])
		self.split = config["split"]
		self.page_retrieval = config["page_retrieval"].lower()
		assert(self.page_retrieval in ["oracle", "concat", "logits", "custom", "maxconf", "anyconf", "maxconfpage", "anyconfpage", "majorpage", "weightmajorpage", "anyconforacle"])

		self.use_images = config.get("use_images", False)
		self.get_raw_ocr_data = config.get("get_raw_ocr_data", False)
		self.max_pages = config.get("max_pages", 1)

	def __len__(self):
		return len(self.dataset)

	def sample(
			self,
			idx: int = None,
			question_id: int = None
	) -> Dict[str, Any]:

		if idx is not None:
			return self.__getitem__(idx)

		if question_id is not None:
			for idx in range(self.__len__()):
				record = self.dataset[idx]
				if record["question_id"] == question_id:
					return self.__getitem__(idx)
				
			raise ValueError(f"Question ID {question_id} not found in the dataset.")

		idx = random.randint(0, self.__len__())
		return self.__getitem__(idx)

	def __getitem__(self, idx: int) -> Dict[str, Any]:
		start_time = time()
		record = self.dataset[idx]

		question = record["questions"]
		if self.split == "train":
			answers = [record.get("labels", "").lower()]
		else:
			answers = record.get("answers", [""])
			if answers is None:
				answers = [""]
			else:
				answers = list(set(answer.lower() for answer in answers))
		answer_page_idx = 0
		num_pages = len(record["ocr_tokens"])

		context = []
		for page_ix in range(num_pages):
			context.append(" ".join(record["ocr_tokens"][page_ix]))

		def ensure_portrait_orientation(img):
			was_rotated = False
			if img.width > img.height:
				img = img.rotate(270, expand=True)
				was_rotated = True
			return img, was_rotated

		rotated_pages = []
		images = None

		if self.use_images:
			image_names = ["" for _ in range(num_pages)]
			images = []
			for i, img in enumerate(record["images"]):
				img_obj, rotated = ensure_portrait_orientation(Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
				images.append(img_obj)
				if rotated:
					rotated_pages.append(i)

		if self.get_raw_ocr_data:
			words = []
			boxes = record["ocr_boxes"].copy()
			for p in range(num_pages):
				words.append([word.lower() for word in record["ocr_tokens"][p]])
				# Transform boxes for rotated pages
				if p in rotated_pages:
					for i, box in enumerate(boxes[p]):
						xmin, ymin, xmax, ymax = box
						boxes[p][i] = [1 - ymax, xmin, 1 - ymin, xmax]

		start_idxs, end_idxs = 0, 0

		sample_info = {
			"question_id": record["question_id"] if "question_id" in record else 0,
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


class DUDE_Raw:
	def __init__(self, config, split):
		self.config = config
		self.split = split

		self.max_pages = config.get("max_pages", 99999) if split == "train" else 99999

	def format_data(self, sample):
		sample = {k: v[0] for k,v in sample.items()}
		new_sample = {
			"questions": [],
			"images": [],
			"ocr_tokens": [],
			"ocr_boxes": []
		}

		n_pages = len(sample["images"])
		images = []
		for image in sample["images"]:
			image = Image.open(io.BytesIO(image))
			image_size = image.size
			scale = 1024 / max(image_size)
			image_size = (int(image_size[0] * scale), int(image_size[1] * scale))
			image = image.resize(image_size)
			images.append(image)

		for i in range(len(sample["questions"])):
			answer_page = random.randint(0, n_pages-1) # Since DUDE does not provide the answer page, we randomly select one
			if n_pages <= self.max_pages:
				first_page, last_page = 0, n_pages

			else:
				first_page_lower_bound = max(0, answer_page-self.max_pages+1)
				first_page_upper_bound = answer_page
				first_page = random.randint(first_page_lower_bound, first_page_upper_bound)
				last_page = first_page + self.max_pages

				if last_page > n_pages:
					last_page = n_pages
					first_page = last_page-self.max_pages

			question = sample["questions"][i]["question"]

			if self.split != "train":
				new_sample["answers"] = new_sample.get("answers", []) + [sample["questions"][i]["answers"]] if "answers" in sample["questions"][i] else []
				new_sample["question_id"] = new_sample.get("question_id", []) + [sample["questions"][i]["question_id"]]
			else:
				new_sample["labels"] = new_sample.get("labels", []) + [random.choice(sample["questions"][i]["answers"])]
			
			new_sample["questions"].append(question)
			new_sample["images"].append(images[first_page:last_page])
			new_sample["ocr_boxes"].append(sample['ocr_boxes'][first_page:last_page])
			new_sample["ocr_tokens"].append(sample['ocr_tokens'][first_page:last_page])       

		return new_sample

def build_dude(config, split):
	dude = DUDE_Raw(config, split)

	dataset_length = {
		"train": 23715,
		"val": 5187,
		"test": 11395
	}
	
	if True:#split != "train":
		# Check if preprocessed dataset exists
		preprocessed_path = os.path.join(config["preprocessed_dir"], "preprocessed", f"DUDE_{split}")
		if os.path.exists(preprocessed_path):
			print(f"Loading preprocessed DUDE dataset from {preprocessed_path}...")
			dataset = load_from_disk(preprocessed_path)
		else:
			# Load and preprocess dataset
			print("Mapping DUDE dataset...")
			dataset = load_dataset(os.path.join(config["data_dir"], "DUDE"), split=split, streaming=False)
			dataset = dataset.map(
				dude.format_data, 
				remove_columns=["images_id"], 
				batched=True, 
				batch_size=1,
				num_proc=30,
			)
			
			# Save preprocessed dataset
			os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
			dataset.save_to_disk(preprocessed_path)
	# else:
	# 	dataset = load_dataset(os.path.join(config["data_dir"], "DUDE"), split=split, streaming=True)
	# 	dataset = dataset.map(
	# 		dude.format_data,
	# 		# remove_columns=["ocr_tokens", "ocr_boxes", "images_id"],
	# 		batched=True,
	# 		batch_size=1
	# 	)

	dataset.num_samples = dataset_length[split]
	dataset.name = "DUDE"

	return dataset


class DUDE_NoisePages(DUDE):
	"""
	DUDE dataset with noise pages functionality.
	For each sample, adds noise pages from random documents within the same dataset,
	excluding the current document.
	"""

	def __init__(
			self,
			config: dict,
	):
		super().__init__(config)
		
		self.noise_pages = config.get("noise_pages", 0)
		self.noise_seed = config.get("noise_seed", 42)
		self.rng = random.Random(self.noise_seed)
		self.mix_noise_pages = config.get("mix_noise_pages", False)

	def _get_document_id(self, record: dict) -> str:
		"""
		Generate a unique document identifier for a DUDE record.
		Since DUDE doesn't have explicit document IDs, we'll use the question_id
		or create one based on the record index.
		"""
		# Use question_id if available, otherwise use the record's hash
		if "question_id" in record and record["question_id"] is not None:
			return str(record["question_id"])
		else:
			# Fallback: use a hash of the first few OCR tokens to identify the document
			if record["ocr_tokens"] and len(record["ocr_tokens"]) > 0 and len(record["ocr_tokens"][0]) > 0:
				first_tokens = " ".join(record["ocr_tokens"][0][:5])  # First 5 tokens
				return str(hash(first_tokens))
			else:
				return str(hash(str(record)))

	def _get_noise_records(self, current_doc_id: str, num_noise_pages: int) -> List[tuple]:
		"""
		Sample noise records from the current dataset, excluding the current document.
		
		Args:
			current_doc_id: The document ID of the current sample
			num_noise_pages: Number of noise pages to sample
			
		Returns:
			List of tuples containing (noise_record, page_index)
		"""
		# Find all records with different document IDs
		candidate_records = []
		for i, rec in tqdm(enumerate(self.dataset)):
			if self._get_document_id(rec) != current_doc_id:
				candidate_records.append(rec)
		
		if len(candidate_records) < num_noise_pages:
			# If we don't have enough different documents, sample with replacement
			noise_records = self.rng.choices(candidate_records, k=num_noise_pages)
		else:
			# Sample without replacement
			noise_records = self.rng.sample(candidate_records, num_noise_pages)
		
		# For each selected document, also select a random page
		noise_records_with_pages = []
		for nrec in noise_records:
			# Select a random page from this document
			num_pages_in_doc = len(nrec["ocr_tokens"])
			if num_pages_in_doc > 0:
				random_page_idx = self.rng.randint(0, num_pages_in_doc - 1)
				noise_records_with_pages.append((nrec, random_page_idx))
		
		return noise_records_with_pages

	def __getitem__(self, idx: int) -> Dict[str, Any]:
		start_time = time()
		record = self.dataset[idx]

		question = record["questions"]
		if self.split == "train":
			answers = [record.get("labels", "").lower()]
		else:
			answers = record.get("answers", [""])
			if answers is None:
				answers = [""]
			else:
				answers = list(set(answer.lower() for answer in answers))
		answer_page_idx = 0
		num_pages = len(record["ocr_tokens"])

		context = []
		for page_ix in range(num_pages):
			context.append(" ".join(record["ocr_tokens"][page_ix]))

		def ensure_portrait_orientation(img):
			was_rotated = False
			if img.width > img.height:
				img = img.rotate(270, expand=True)
				was_rotated = True
			return img, was_rotated

		rotated_pages = []
		images = None

		if self.use_images:
			image_names = ["" for _ in range(num_pages)]
			images = []
			for i, img in enumerate(record["images"]):
				img_obj, rotated = ensure_portrait_orientation(Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
				images.append(img_obj)
				if rotated:
					rotated_pages.append(i)

		if self.get_raw_ocr_data:
			words = []
			boxes = record["ocr_boxes"].copy()
			for p in range(num_pages):
				words.append([word.lower() for word in record["ocr_tokens"][p]])
				# Transform boxes for rotated pages
				if p in rotated_pages:
					for i, box in enumerate(boxes[p]):
						xmin, ymin, xmax, ymax = box
						boxes[p][i] = [1 - ymax, xmin, 1 - ymin, xmax]

		# Add noise pages
		if self.noise_pages > 0:
			current_doc_id = self._get_document_id(record)
			noise_records = self._get_noise_records(current_doc_id, self.noise_pages)
			
			for nrec, random_page_idx in noise_records:
				# Append OCR text from the random page
				if random_page_idx < len(nrec["ocr_tokens"]):
					context.append(" ".join(nrec["ocr_tokens"][random_page_idx]))
					
					# Append boxes/words if requested
					if self.get_raw_ocr_data:
						words.append([word.lower() for word in nrec["ocr_tokens"][random_page_idx]])
						if random_page_idx < len(nrec["ocr_boxes"]):
							boxes.append(nrec["ocr_boxes"][random_page_idx])
						else:
							# Fallback: empty boxes if page doesn't exist
							boxes.append([])
					
					# Append images if requested
					if self.use_images and random_page_idx < len(nrec["images"]):
						noise_img = nrec["images"][random_page_idx]
						noise_img_obj, noise_rotated = ensure_portrait_orientation(
							Image.open(io.BytesIO(noise_img["bytes"])).convert("RGB")
						)
						images.append(noise_img_obj)
						image_names.append("")  # No specific name for noise images
						
						# Handle rotation for noise page boxes if needed
						if noise_rotated and self.get_raw_ocr_data:
							noise_boxes = boxes[-1]  # Get the boxes we just added
							for i, box in enumerate(noise_boxes):
								if len(box) == 4:
									xmin, ymin, xmax, ymax = box
									noise_boxes[i] = [1 - ymax, xmin, 1 - ymin, xmax]

			# --- MIXING LOGIC ---
			if self.mix_noise_pages:
				# Gather all page-wise lists
				page_tuples = list(zip(context,
					words if self.get_raw_ocr_data else [None]*len(context),
					boxes if self.get_raw_ocr_data else [None]*len(context),
					images if self.use_images else [None]*len(context),
					image_names if self.use_images else [None]*len(context)))
				self.rng.shuffle(page_tuples)
				# Unpack
				context = [t[0] for t in page_tuples]
				if self.get_raw_ocr_data:
					words = [t[1] for t in page_tuples]
					boxes = [t[2] for t in page_tuples]
				if self.use_images:
					images = [t[3] for t in page_tuples]
					image_names = [t[4] for t in page_tuples]

		start_idxs, end_idxs = 0, 0

		sample_info = {
			"question_id": record["question_id"] if "question_id" in record else 0,
			"questions": question,
			"contexts": context,
			"context_page_corresp": None,
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

		if self.get_raw_ocr_data:
			sample_info["words"] = words
			sample_info["boxes"] = boxes
		else:  # Information for extractive models
			sample_info["start_indxs"] = start_idxs
			sample_info["end_indxs"] = end_idxs

		return sample_info
