import transformers
from src.rag_vt5 import RAGVT5
from src.SP_DocVQA import SPDocVQA
from transformers import get_scheduler


def build_optimizer(model, length_train_loader, config):
	optimizer_class = getattr(transformers, "AdamW")
	optimizer = optimizer_class(model.model.parameters(), lr=float(config["lr"]))
	num_training_steps = config["train_epochs"] * length_train_loader
	lr_scheduler = get_scheduler(
		name="linear", optimizer=optimizer, num_warmup_steps=config["warmup_iterations"], num_training_steps=num_training_steps
	)
	return optimizer, lr_scheduler


def build_model(config):
	model = RAGVT5(config)
	model.to(model.device)
	return model


def build_dataset(config, split):
	dataset_kwargs = {
		"get_raw_ocr_data": True,
		"use_images": True,
	}
	return SPDocVQA(config["imdb_dir"], config["images_dir"], split, dataset_kwargs)