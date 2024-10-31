import os
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.RAG_VT5 import RAGVT5
from src.metrics import Evaluator
from src.logger import Logger
from src.utils import seed_everything, load_config
from src.build_utils import build_model, build_dataset, build_optimizer
from src.MP_DocVQA import mpdocvqa_collate_fn
from eval import evaluate
from src.checkpoint import save_model
from typing import Any

def train_epoch(
		data_loader: DataLoader,
		model: RAGVT5,
		optimizer: Any,
		lr_scheduler: Any,
		evaluator: Evaluator,
		logger: Logger,
		**kwargs
):
	model.train()

	for batch_idx, batch in enumerate(tqdm(data_loader)):
		gt_answers = batch["answers"]
		outputs, pred_answers, pred_answer_pages, pred_answers_conf, _ = model.forward(
			batch,
			return_pred_answer=True,
			return_retrieval=False,
			chunk_num=kwargs.get("chunk_num", 5),
			chunk_size=kwargs.get("chunk_size", 30),
			overlap=kwargs.get("overlap", 0),
			include_surroundings=kwargs.get("include_surroundings", 10)
		)
		loss = outputs.loss + outputs.ret_loss if hasattr(outputs, "ret_loss") else outputs.loss

		loss.backward()
		optimizer.step()
		lr_scheduler.step()

		optimizer.zero_grad()

		metric = evaluator.get_metrics(gt_answers, pred_answers)

		batch_acc = np.mean(metric["accuracy"])
		batch_anls = np.mean(metric["anls"])

		log_dict = {
			"Train/Batch loss": outputs.loss.item(),
			"Train/Batch Accuracy": batch_acc,
			"Train/Batch ANLS": batch_anls,
			"lr": optimizer.param_groups[0]["lr"]
		}

		if hasattr(outputs, "ret_loss"):
			log_dict["Train/Batch retrieval loss"] = outputs.ret_loss.item()

		if "answer_page_idx" in batch and None not in batch["answer_page_idx"] and pred_answer_pages is not None:
			ret_metric = evaluator.get_retrieval_metric(batch.get("answer_page_idx", None), pred_answer_pages)
			batch_ret_prec = np.mean(ret_metric)
			log_dict["Train/Batch Ret. Prec."] = batch_ret_prec

		logger.logger.log(log_dict, step=logger.current_epoch * logger.len_dataset + batch_idx)

def train(model, **kwargs):

	epochs = kwargs["train_epochs"]
	seed_everything(kwargs["seed"])

	evaluator = Evaluator(case_sensitive=False)
	logger = Logger(config=kwargs)
	logger.log_model_parameters(model)

	print("Building dataset...")
	train_dataset = build_dataset(config, split="train", size=kwargs.get("train_size", 1.0))
	val_dataset   = build_dataset(config, split="val", size=kwargs.get("val_size", 1.0))

	train_data_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=mpdocvqa_collate_fn)
	val_data_loader   = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=mpdocvqa_collate_fn)

	logger.len_dataset = len(train_data_loader)
	optimizer, lr_scheduler = build_optimizer(model, length_train_loader=len(train_data_loader), config=kwargs)

	if kwargs.get("eval_start", False):
		logger.current_epoch = -1
		eval_res = evaluate(val_data_loader, model, evaluator, return_scores_by_sample=False, return_answers=False, save_results=False, **kwargs)
		accuracy = np.mean(eval_res["accuracy"])
		anls = np.mean(eval_res["anls"])
		retrieval_precision = np.mean(eval_res["retrieval_precision"])
		avg_chunk_score = np.mean(eval_res["chunk_score"])
		is_updated = evaluator.update_global_metrics(accuracy, anls, -1)
		logger.log_val_metrics(accuracy, anls, retrieval_precision, avg_chunk_score, update_best=is_updated)

	for epoch_ix in range(epochs):
		logger.current_epoch = epoch_ix
		train_epoch(train_data_loader, model, optimizer, lr_scheduler, evaluator, logger, **kwargs)
		eval_res = evaluate(val_data_loader, model, evaluator, return_scores_by_sample=False, return_answers=False, save_results=False, **kwargs)
		print(f"Epoch {epoch_ix} completed")
		accuracy = np.mean(eval_res["accuracy"])
		anls = np.mean(eval_res["anls"])
		retrieval_precision = np.mean(eval_res["retrieval_precision"])
		avg_chunk_score = np.mean(eval_res["chunk_score"])
		is_updated = evaluator.update_global_metrics(accuracy, anls, epoch_ix)
		logger.log_val_metrics(accuracy, anls, retrieval_precision, avg_chunk_score, update_best=is_updated)
		save_model(model, epoch_ix, update_best=is_updated, **kwargs)
		print("Model saved")

if __name__ == "__main__":
    # Prepare model and dataset
	args = {
		"model": "RAGVT5", # RAGVT5, HiVT5
		"dataset": "MP-DocVQA",
		"embed_model": "BGE", # BGE, VT5
		"page_retrieval": "Concat", # Oracle / Concat / Logits / Maxconf / Custom (HiVT5 only)
		"chunk_num": 10,
		"chunk_size": 60,
		"overlap": 10,
		"include_surroundings": 0,
		"visible_devices": "5",
		"save_name_append": "no-token",
		"add_sep_token": True,
	}
	os.environ["CUDA_VISIBLE_DEVICES"] = args["visible_devices"]
	args = argparse.Namespace(**args)
	config = load_config(args)
	config["train_size"] = 1.0
	config["val_size"] = 1.0
	print("Building model...")
	model = build_model(config)
	model.to(config["device"])
	train(model, **config)
