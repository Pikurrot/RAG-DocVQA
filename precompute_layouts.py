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

def precompute_layouts(dataloader, layout_model, config):
	output_dir = config["output_dir"]
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	for images, image_paths in tqdm(dataloader):
		doc_ids = [os.path.basename(p).split("_")[0] for p in image_paths]
		page_nums = [os.path.basename(p).split("_")[1].split(".")[0] for p in image_paths]
		out_filenames = [os.path.join(output_dir, doc_id)+".npz" for doc_id in doc_ids]
		layout_info, _ = layout_model(images)
		# Turn lists into arrays
		for elem in layout_info:
			elem["boxes"] = np.array(elem["boxes"])
			elem["labels"] = np.array(elem["labels"])
		# Save layouts
		for page_layout, out_filename, page_num in zip(layout_info, out_filenames, page_nums):
			page_layout_info = {page_num: page_layout}
			if os.path.exists(out_filename):
				with np.load(out_filename, allow_pickle=True) as f:
					existing_layout_info = dict(f)
				page_layout_info.update(existing_layout_info)
			np.savez(out_filename, **page_layout_info)

def worker_process(gpu_id, num_gpus, config):
	# Set the GPU for this process
	os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
	
	# Build dataset and then partition by taking every num_gpus-th element starting from gpu_id
	dataset = ImageDataset(config["images_dir"])
	indices = list(range(len(dataset)))
	# Partition the dataset: each process gets a slice based on modulo
	subset_indices = indices[gpu_id::num_gpus]
	subset = Subset(dataset, subset_indices)
	
	dataloader = DataLoader(subset, batch_size=config["batch_size"], shuffle=False, collate_fn=image_lst_collate_fn)
	layout_model = LayoutModel(config)  # The model should load on the chosen GPU by reading CUDA_VISIBLE_DEVICES
	
	precompute_layouts(dataloader, layout_model, config)

def main():
	args = {
		"model": "RAGVT5",
		"dataset": "MP-DocVQA",
		"batch_size": 10,
		"layout_batch_size": 10,
		"layout_model_weights": "cmarkea/dit-base-layout-detection",
		"use_layout_labels": True,
		"output_dir": "/data3fast/users/elopez/data/images_layouts",
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
	
	# Get the list of GPUs to use from the config
	visible_devices = [int(x) for x in config["visible_devices"].split(",")]
	num_gpus = len(visible_devices)
	
	mp.spawn(worker_process, args=(num_gpus, config), nprocs=num_gpus, join=True)

if __name__ == "__main__":
	main()
