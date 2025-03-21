import transformers
from src.RAGVT5 import RAGVT5
from src.HiVT5 import Proxy_HiVT5
from src.MP_DocVQA import MPDocVQA
from src.Infographics import Infographics
from src.DUDE import DUDE
from transformers import get_scheduler
from typing import Literal

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
	device = config.get("device", "cuda")
	if config["model_name"] == "RAGVT5":
		model = RAGVT5(config)
	elif config["model_name"] == "Hi-VT5":
		model = Proxy_HiVT5(config)
	model.to(device)
	return model

def build_dataset(
		config: dict,
		split: Literal["train", "val", "test"],
		size: float=1.0,
		**kwargs
):
	dataset_config = config.copy()
	dataset_config.update({
		"get_raw_ocr_data": True,
		"use_images": True,
		"size": size,
		"split": split
	})
	if config["model_name"] == "Hi-VT5":
		dataset_config.update({
			"max_pages": config.get("max_pages", 1),
			"hierarchical_method": True
		})
	dataset_config.update(kwargs)
	if config["dataset_name"].lower() == "mp-docvqa":
		dataset = MPDocVQA(dataset_config)
	elif config["dataset_name"].lower() == "infographics":
		dataset = Infographics(dataset_config)
	elif config["dataset_name"].lower() == "dude":
		dataset = DUDE(dataset_config)
	return dataset
