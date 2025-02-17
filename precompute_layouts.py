import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torch.multiprocessing as mp
from src.utils import load_config
from src._modules import LayoutModel

def image_lst_collate_fn(batch):
	images, image_paths = zip(*batch)
	return images, image_paths

class ImageDataset(Dataset):
	def __init__(self, root, transform=None):
		self.root = root
		self.transform = transform
		self.image_files = [
			os.path.join(root, f)
			for f in os.listdir(root)
			if f.lower().endswith((".png", ".jpg", ".jpeg"))
		]
	
	def __len__(self):
		return len(self.image_files)
	
	def __getitem__(self, index):
		image_path = self.image_files[index]
		image = Image.open(image_path).convert("RGB")
		if self.transform:
			image = self.transform(image)
		return image, image_path

def precompute_layouts_aggregated(dataloader, layout_model, config):
	results = {}
	for images, image_paths in tqdm(dataloader):
		# Use the image filename (without extension) as key
		keys = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
		layout_info, _ = layout_model(images)
		for key, elem in zip(keys, layout_info):
			# Convert lists to numpy arrays
			elem["boxes"] = np.array(elem["boxes"])
			elem["labels"] = np.array(elem["labels"])
			results[key] = elem
	return results

def worker_process(gpu_id, num_gpus, config, shared_dict):
	# Set the GPU for this process
	os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
	
	# Build dataset and assign every num_gpus-th image to this worker
	dataset = ImageDataset(config["images_dir"])
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
	results = precompute_layouts_aggregated(dataloader, layout_model, config)
	
	# Update the shared dictionary with this process's results
	shared_dict.update(results)

def main():
	args = {
		"model": "RAGVT5",
		"layout_model": "YOLO", # YOLO, DIT
		"dataset": "MP-DocVQA",
		"batch_size": 10,
		"layout_batch_size": 10,
		"layout_model_weights": "cmarkea/dit-base-layout-detection",
		"use_layout_labels": True,
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
	out_filename = os.path.join(output_dir, "images_layouts.npz")
	np.savez_compressed(out_filename, **aggregated_results)
	print(f"Merged layout file saved to {out_filename}")

if __name__ == "__main__":
	main()
