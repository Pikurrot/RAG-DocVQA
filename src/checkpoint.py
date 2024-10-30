import os
from src.utils import save_yaml
from src.RAG_VT5 import RAGVT5

def save_model(
		model: RAGVT5,
		epoch: int,
		update_best: bool=False,
		**kwargs
):
	model_name = "model__{:d}.ckpt".format(epoch)
	save_name_append = kwargs.get("save_name_append", "")
	save_dir = os.path.join(kwargs["save_dir"], "checkpoints", "{:s}_{:s}_{:s}{:s}".format(
		kwargs["model_name"].lower(),
		kwargs.get("page_retrieval", "").lower(),
		kwargs["dataset_name"].lower(),
		"_"+save_name_append if save_name_append else ""
	))
	model.generator.save_pretrained(os.path.join(save_dir, model_name))

	tokenizer = model.generator.tokenizer if hasattr(model.generator, "tokenizer") else model.generator.processor if hasattr(model.generator, "processor") else None
	if tokenizer is not None:
		tokenizer.save_pretrained(os.path.join(save_dir, model_name))

	if hasattr(model.generator, "visual_embedding"):
		model.generator.visual_embedding.feature_extractor.save_pretrained(os.path.join(save_dir, model_name))

	save_yaml(os.path.join(save_dir, model_name, "experiment_config.yml"), kwargs)

	if update_best:
		model.generator.save_pretrained(os.path.join(save_dir, "best.ckpt"))
		if tokenizer is not None:
			tokenizer.save_pretrained(os.path.join(save_dir, "best.ckpt"))
		save_yaml(os.path.join(save_dir, "best.ckpt", "experiment_config.yml"), kwargs)

def load_model(
		base_model: RAGVT5,
		ckpt_name: str,
		**kwargs
):
	load_dir = kwargs["save_dir"]
	base_model.generator.from_pretrained(os.path.join(load_dir, ckpt_name))
