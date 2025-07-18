import transformers
from src.RAGVT5 import RAGVT5
from src.HiVT5 import Proxy_HiVT5
from src.RAGPix2Struct import RAGPix2Struct
from src.MP_DocVQA import MPDocVQA, MPDocVQA_NoisePagesv2
from src.Infographics import Infographics
from src.DUDE import DUDE, DUDE_NoisePages, create_balanced_nac_dataset
from src.SP_DocVQA import SPDocVQA
from src.MMLongBenchDoc import MMLongBenchDoc
from transformers import get_scheduler
from typing import Literal, Union

def build_optimizer(
		model: Union[RAGVT5, RAGPix2Struct],
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
	elif config["model_name"] == "RAGPix2Struct":
		model = RAGPix2Struct(config)
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
	elif config["dataset_name"].lower() == "mp-docvqa-noise":
		dataset = MPDocVQA_NoisePagesv2(dataset_config)
	elif config["dataset_name"].lower() == "infographics":
		dataset = Infographics(dataset_config)
	elif config["dataset_name"].lower() == "dude":
		dataset = DUDE(dataset_config)
		if config.get("balance_nac_dataset", False):
			dataset = create_balanced_nac_dataset(dataset, config.get("nac_dataset_ratio", 0.5))
	elif config["dataset_name"].lower() == "dude-noise":
		dataset = DUDE_NoisePages(dataset_config)
	elif config["dataset_name"].lower() == "sp-docvqa":
		dataset = SPDocVQA(dataset_config)
	elif config["dataset_name"].lower() == "mmlongbenchdoc":
		dataset = MMLongBenchDoc(dataset_config)
	return dataset
