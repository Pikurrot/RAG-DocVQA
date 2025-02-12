import os
from src.utils import save_yaml
from src.RAGVT5 import RAGVT5

def save_model(
		model: RAGVT5,
		epoch: int,
		config: dict,
		update_best: bool=False,
):
	model_name = "model__{:d}.ckpt".format(epoch)
	save_name_append = config.get("save_name_append", "")
	save_dir = os.path.join(config["checkpoint_dir"], "{:s}_{:s}_{:s}{:s}".format(
		config["model_name"].lower(),
		config.get("page_retrieval", "").lower(),
		config["dataset_name"].lower(),
		"_"+save_name_append if save_name_append else ""
	))
	model.generator.save_pretrained(os.path.join(save_dir, model_name))

	tokenizer = model.generator.tokenizer if hasattr(model.generator, "tokenizer") else model.generator.processor if hasattr(model.generator, "processor") else None
	if tokenizer is not None:
		tokenizer.save_pretrained(os.path.join(save_dir, model_name))

	if hasattr(model.generator, "visual_embedding"):
		model.generator.visual_embedding.feature_extractor.save_pretrained(os.path.join(save_dir, model_name))

	save_yaml(os.path.join(save_dir, model_name, "experiment_config.yml"), config)

	if update_best:
		model.generator.save_pretrained(os.path.join(save_dir, "best.ckpt"))
		if tokenizer is not None:
			tokenizer.save_pretrained(os.path.join(save_dir, "best.ckpt"))
		save_yaml(os.path.join(save_dir, "best.ckpt", "experiment_config.yml"), config)

def load_model(
		base_model: RAGVT5,
		ckpt_name: str,
		**kwargs
):
	load_dir = kwargs["save_dir"]
	base_model.generator.from_pretrained(os.path.join(load_dir, ckpt_name))
