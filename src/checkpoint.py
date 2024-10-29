import os
from utils import save_yaml
from src.RAG_VT5 import RAGVT5

def save_model(
        model: RAGVT5,
        epoch: int,
        update_best: bool=False,
        **kwargs
):
    save_dir = os.path.join(kwargs["save_dir"], "checkpoints", "{:s}_{:s}_{:s}".format(kwargs["model_name"].lower(), kwargs.get("page_retrieval", "").lower(), kwargs["dataset_name"].lower()))
    model.generator.save_pretrained(os.path.join(save_dir, "model__{:d}.ckpt".format(epoch)))

    tokenizer = model.generator.tokenizer if hasattr(model, "tokenizer") else model.generator.processor if hasattr(model, "processor") else None
    if tokenizer is not None:
        tokenizer.save_pretrained(os.path.join(save_dir, "model__{:d}.ckpt".format(epoch)))

    if hasattr(model.generator, "visual_embedding"):
        model.generator.visual_embedding.feature_extractor.save_pretrained(os.path.join(save_dir, "model__{:d}.ckpt".format(epoch)))

    save_yaml(os.path.join(save_dir, "model__{:d}.ckpt".format(epoch), "experiment_config.yml"), kwargs)

    if update_best:
        model.generator.save_pretrained(os.path.join(save_dir, "best.ckpt"))
        tokenizer.save_pretrained(os.path.join(save_dir, "best.ckpt"))
        save_yaml(os.path.join(save_dir, "best.ckpt", "experiment_config.yml"), kwargs)

def load_model(
        base_model: RAGVT5,
        ckpt_name: str,
        **kwargs
):
    load_dir = kwargs["save_dir"]
    base_model.generator.from_pretrained(os.path.join(load_dir, ckpt_name))
