import os
import io
import random
from PIL import Image
from datasets import load_dataset, load_from_disk

class DUDE:
	def __init__(self, config, split):
		self.config = config
		self.split = split

		self.max_pages = config.get("max_pages", 99999) if split == "train" else 99999

	def format_data(self, sample):
		sample = {k: v[0] for k,v in sample.items()}
		new_sample = {
			"questions": [],
			"images": [],
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

		return new_sample

def build_dude(config, split):
	dude = DUDE(config, split)

	dataset_length = {
		"train": 23715,
		"val": 5187,
		"test": 11395
	}
	
	if split != "train":
		# Check if preprocessed dataset exists
		preprocessed_path = os.path.join(config["data_dir"], "preprocessed", f"DUDE_{split}")
		if os.path.exists(preprocessed_path):
			dataset = load_from_disk(preprocessed_path)
		else:
			# Load and preprocess dataset
			dataset = load_dataset(os.path.join(config["data_dir"], "DUDE"), split=split, streaming=False)
			dataset = dataset.map(
				dude.format_data, 
				remove_columns=["ocr_tokens", "ocr_boxes", "images_id"], 
				batched=True, 
				batch_size=1
			)
			
			# Save preprocessed dataset
			os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
			dataset.save_to_disk(preprocessed_path)
	else:
		dataset = load_dataset(os.path.join(config["data_dir"], "DUDE"), split=split, streaming=True)
		dataset = dataset.map(
			dude.format_data,
			remove_columns=["ocr_tokens", "ocr_boxes", "images_id"],
			batched=True,
			batch_size=1
		)

	dataset.num_samples = dataset_length[split]
	dataset.name = "DUDE"

	return dataset
