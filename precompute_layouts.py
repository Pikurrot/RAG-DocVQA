import os
import argparse
import numpy as np
import torch.multiprocessing as mp
import json
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from src.utils import load_config
from src._modules import LayoutModel, S2Chunker, Embedder

def image_lst_collate_fn(batch):
	return  zip(*batch)

class ImageDataset(Dataset):
	def __init__(self, root, transform=None, use_ocr=False):
		self.root = root
		self.image_path = os.path.join(root, "images")
		self.ocr_path = os.path.join(root, "ocr")
		self.transform = transform
		self.use_ocr = use_ocr
		basenames = [
			os.path.basename(f).split(".")[0]
			for f in os.listdir(self.image_path)
			if f.lower().endswith((".png", ".jpg", ".jpeg"))
		]
		self.image_files = [
			os.path.join(self.image_path, f"{basename}.jpg")
			for basename in basenames
		]
		if use_ocr:
			self.ocr_files = [
				os.path.join(self.ocr_path, f"{basename}.json")
				for basename in basenames
			]
		else:
			self.ocr_files = None
			self.pages_info = None

	def __len__(self):
		return len(self.image_files)

	def __getitem__(self, index):
		image_path = self.image_files[index]
		image = Image.open(image_path).convert("RGB")
		if self.transform:
			image = self.transform(image)
		if self.use_ocr:
			ocr_file = self.ocr_files[index]
			with open(ocr_file, "r") as f:
				data = json.load(f)
			page_info = {
				"ocr_tokens": [],
				"ocr_normalized_boxes": []
			}
			if "WORD" in data:
				for word in data["WORD"]:
					page_info["ocr_tokens"].append(word["Text"])
					page_info["ocr_normalized_boxes"].append([
						word["Geometry"]["BoundingBox"]["Left"],
						word["Geometry"]["BoundingBox"]["Top"],
						word["Geometry"]["BoundingBox"]["Left"] + word["Geometry"]["BoundingBox"]["Width"],
						word["Geometry"]["BoundingBox"]["Top"] + word["Geometry"]["BoundingBox"]["Height"]
					])
			else:
				page_info = None
			return image, image_path, page_info
		return image, image_path, None

def precompute_layouts_aggregated(dataloader, layout_model, clusterer, config):
	results = {}
	for images, image_paths, pages_info in tqdm(dataloader):
		# Use the image filename (without extension) as key
		keys = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
		layout_info, _ = layout_model(images)
		if config["cluster_layouts"]:
			try:
				batch_clusters = clusterer(layout_info, pages_info)
			except Exception as e:
				print(image_paths)
				raise e
		else:
			batch_clusters = [np.full(len(page_layout_info), -1) for page_layout_info in layout_info]
		for key, elem, clusters in zip(keys, layout_info, batch_clusters):
			# Convert lists to numpy arrays
			elem["boxes"] = np.array(elem["boxes"])
			elem["labels"] = np.array(elem["labels"])
			elem["clusters"] = clusters
			results[key] = elem
	return results

def worker_process(gpu_id, num_gpus, config, shared_dict):
	# Set the GPU for this process
	os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
	
	# Build dataset and assign every num_gpus-th image to this worker
	dataset = ImageDataset(os.path.dirname(config["images_dir"]), use_ocr=config["cluster_layouts"])
	indices = list(range(len(dataset)))
	subset_indices = indices[gpu_id::num_gpus]
	subset = Subset(dataset, subset_indices)
	
	dataloader = DataLoader(
		subset,
		batch_size=config["batch_size"],
		shuffle=False,
		collate_fn=image_lst_collate_fn
	)
	layout_model = LayoutModel(config)  # Loads on the chosen GPU
	embedder = Embedder(config)
	clusterer = S2Chunker(config, embedder=embedder)
	results = precompute_layouts_aggregated(dataloader, layout_model, clusterer, config)
	
	# Update the shared dictionary with this process's results
	shared_dict.update(results)

def main():
	args = {
		"model": "RAGVT5",
		"embed_model": "BGE",
		"layout_model": "DIT", # YOLO, DIT
		"dataset": "MP-DocVQA",
		"batch_size": 10,
		"layout_batch_size": 10,
		"chunk_size": 60,
		"chunk_size_tol": 0.2,
		"embed_weights": "/data3fast/users/elopez/models/bge-finetuned-2/checkpoint-820",
		"layout_model_weights": "cmarkea/dit-base-layout-detection",
		"use_layout_labels": True,
		"cluster_layouts": True,
		"cluster_mode": "spatial+semantic", # spatial, spatial+semantic
		"calculate_n_clusters": "best", # heuristic, best
		"output_dir": "/data3fast/users/elopez/data",
	}
	extra_args = {
		"visible_devices": "1,2,3,4,5",
		"data_size": 1.0,
		"compute_stats": False,
		"compute_stats_examples": False,
		"n_stats_examples": 0
	}
	args.update(extra_args)
	args = argparse.Namespace(**args)
	config = load_config(args)
	
	manager = mp.Manager()
	shared_dict = manager.dict()
	
	visible_devices = [int(x) for x in config["visible_devices"].split(",")]
	num_gpus = len(visible_devices)
	
	mp.spawn(worker_process, args=(num_gpus, config, shared_dict), nprocs=num_gpus, join=True)
	
	# After all workers finish, convert the shared dict to a regular dict and save
	aggregated_results = dict(shared_dict)
	output_dir = config["output_dir"]
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	out_filename = os.path.join(output_dir, "images_layouts_dit_s2.npz")
	np.savez_compressed(out_filename, **aggregated_results)
	print(f"Merged layout file saved to {out_filename}")

if __name__ == "__main__":
	main()
