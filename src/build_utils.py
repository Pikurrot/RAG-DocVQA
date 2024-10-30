import transformers
from src.RAG_VT5 import RAGVT5
from src.HiVT5 import Proxy_HiVT5
from src.MP_DocVQA import MPDocVQA
from transformers import get_scheduler
from typing import Any, Literal

def build_optimizer(
		model: RAGVT5,
		length_train_loader: int,
		config: dict
):
	optimizer_class = getattr(transformers, "AdamW")
	optimizer = optimizer_class(model.generator.parameters(), lr=float(config["lr"]))
	num_training_steps = config["train_epochs"] * length_train_loader
	lr_scheduler = get_scheduler(
		name="linear", optimizer=optimizer, num_warmup_steps=config["warmup_iterations"], num_training_steps=num_training_steps
	)
	return optimizer, lr_scheduler

def build_model(config: dict) -> RAGVT5:
	if config["model_name"] == "RAGVT5":
		model = RAGVT5(config)
	elif config["model_name"] == "Hi-VT5":
		model = Proxy_HiVT5(config)
	model.to(model.device)
	return model

def build_dataset(
		config: dict,
		split: Literal["train", "val", "test"],
		size: float=1.0
):
	dataset_kwargs = {
		"get_raw_ocr_data": True,
		"use_images": True,
		"size": size
	}
	if config["model_name"] == "Hi-VT5":
		dataset_kwargs.update({
			"max_pages": config.get("max_pages", 1),
			"hierarchical_method": True
		})
	return MPDocVQA(config["imdb_dir"], config["images_dir"], config["page_retrieval"], split, dataset_kwargs)
