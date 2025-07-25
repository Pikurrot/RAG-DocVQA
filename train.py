import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import numpy as np
import datetime
import time
import torch
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.RAGVT5 import RAGVT5
from src.metrics import Evaluator
from src.logger import Logger, LoggerEval
from src.utils import seed_everything, load_config
from src.build_utils import build_model, build_dataset, build_optimizer
from src.MP_DocVQA import mpdocvqa_collate_fn
from eval import evaluate
from src.checkpoint import save_model
from typing import Any

def get_grad_norm(module: torch.nn.Module) -> float:
	total_norm = 0.0
	for p in module.parameters():
		if p.grad is not None:
			total_norm += p.grad.data.norm(2).item() ** 2
	return total_norm ** 0.5

def train_epoch(
		data_loader: DataLoader,
		model: RAGVT5,
		optimizer: Any,
		lr_scheduler: Any,
		evaluator: Evaluator,
		logger: Logger,
		config
):
	model.train()

	# modules = {
	# 	"language_backbone": model.generator.language_backbone,
	# 	"spatial_embedding": model.generator.spatial_embedding,
	# 	"visual_embedding": model.generator.visual_embedding,
	# 	# "layout_embedding": model.generator.layout_embedding
	# }
	modules = {
		"generator": model.generator,
		"not_answerable_classifier": model.not_answerable_classifier if model.use_not_answerable_classifier else None
	}
	print_first_samples = config.get("print_first_samples", 0)

	for batch_idx, batch in enumerate(tqdm(data_loader)):
		gt_answers = batch["answers"]
		if data_loader.dataset.name == "DUDE":
			answer_type = batch["answer_type"] # (bs,)
			not_answerable = [answer_type[i] == "not-answerable" for i in range(len(answer_type))]
			not_answerable_gt = torch.tensor(not_answerable, dtype=torch.float32, device=config["device"]) # (bs,)
		try:
			outputs, pred_answers, pred_answer_pages, _, retrieval = model.forward(
				batch,
				return_pred_answer=True,
				return_retrieval=True  # Changed to True to get retrieval info
			)
			if "not_answerable_probs" in retrieval:
				not_answerable_pred = retrieval["not_answerable_probs"].squeeze(-1)  # (bs,)
			
			# Calculate main loss
			loss = outputs.loss + outputs.ret_loss if hasattr(outputs, "ret_loss") else outputs.loss
			
			# Add NotAnswerableClassifier loss if applicable
			if (model.use_not_answerable_classifier and 
				model.train_not_answerable_classifier and 
				data_loader.dataset.name == "DUDE" and
				"not_answerable_probs" in retrieval):
				
				nac_loss_weight = config.get("nac_loss_weight", 1.0)
				nac_loss = torch.nn.functional.binary_cross_entropy(
					not_answerable_pred, 
					not_answerable_gt
				)
				loss = loss + nac_loss_weight * nac_loss
				
				nac_pred_binary = (not_answerable_pred > 0.5).float()
				nac_accuracy = (nac_pred_binary == not_answerable_gt).float().mean()

			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)

			# Log gradients
			grad_norms = {}
			for name, module in modules.items():
				if module is None:
					continue
				# Only consider modules with parameters that have gradients
				if any(p.grad is not None for p in module.parameters()):
					norm = get_grad_norm(module)
					grad_norms[name] = norm

			optimizer.step()
			lr_scheduler.step()
			optimizer.zero_grad()

			if (print_first_samples > 0 and batch_idx < print_first_samples) or (print_first_samples == -1):
				print(f"Batch {batch_idx}:")
				print("Answers:", gt_answers)
				print("Predicted answers:", pred_answers)
				print("-" * 20)
			metric = evaluator.get_metrics(gt_answers, pred_answers)

			batch_acc = np.mean(metric["accuracy"])
			batch_anls = np.mean(metric["anls"])

			log_dict = {
				"Train/Batch loss": outputs.loss.item(),
				"Train/Batch Accuracy": batch_acc,
				"Train/Batch ANLS": batch_anls,
				"Train/lr": optimizer.param_groups[0]["lr"],
				"Train/Batch Grad Norm": grad_norms,
				# "Train/Batch lm_loss": outputs.lm_loss.item(),
				# "Train/Batch layout_loss": outputs.layout_loss.item(),
				# "Train/layout_embedding_scale": model.generator.layout_embedding_scale.item()
			}

			if hasattr(outputs, "ret_loss"):
				log_dict["Train/Batch retrieval loss"] = outputs.ret_loss.item()

			if "answer_page_idx" in batch and None not in batch["answer_page_idx"] and pred_answer_pages is not None:
				ret_metric = evaluator.get_retrieval_metric(batch.get("answer_page_idx", None), pred_answer_pages)
				batch_ret_prec = np.mean(ret_metric)
				log_dict["Train/Batch Ret. Prec."] = batch_ret_prec

			if model.use_not_answerable_classifier:
				log_dict["Train/Batch NAC Loss"] = nac_loss.item()
				log_dict["Train/Batch NAC Accuracy"] = nac_accuracy.item()

			logger.log(log_dict, step=logger.current_epoch * logger.len_dataset + batch_idx)
		except torch.OutOfMemoryError:
			print("Out of memory warning. Skipping batch.")
			gc.collect()
			torch.cuda.empty_cache()

def train(
		train_data_loader: DataLoader,
		val_data_loader: DataLoader,
		model: RAGVT5,
		evaluator: Evaluator,
		logger_train: Logger,
		logger_eval: LoggerEval,
		config: dict,
		filename: str,
		start_time: float,
):
	epochs = config["train_epochs"]
	seed_everything(config["seed"])
	logger_train.log_model_parameters(model)
	logger_train.len_dataset = len(train_data_loader)
	optimizer, lr_scheduler = build_optimizer(model, length_train_loader=len(train_data_loader), config=config)
	is_updated = True

	if config.get("eval_start", False):
		logger_train.current_epoch = -1
		eval_res = evaluate(
			val_data_loader,
			model, evaluator, logger_eval,
			config,
			filename=filename,
			start_time=start_time
		)
		is_updated = evaluator.update_global_metrics(eval_res["accuracy"], eval_res["anls"], -1)
		logger_train.log_val_metrics(eval_res["accuracy"], eval_res["anls"], eval_res["retrieval_precision"], eval_res["chunk_score"], update_best=is_updated)

	for epoch_ix in range(epochs):
		logger_train.current_epoch = epoch_ix
		train_epoch(
			train_data_loader,
			model, optimizer, lr_scheduler, evaluator, logger_train,
			config
		)
		save_model(model, epoch_ix, config, update_best=is_updated)
		print("Model saved")
		torch.cuda.empty_cache()
		eval_res = evaluate(
			val_data_loader,
			model, evaluator, logger_eval,
			config,
			filename=filename,
			start_time=start_time
		)
		print(f"Epoch {epoch_ix} completed")
		is_updated = evaluator.update_global_metrics(eval_res["accuracy"], eval_res["anls"], epoch_ix)
		logger_train.log_val_metrics(eval_res["accuracy"], eval_res["anls"], eval_res["retrieval_precision"], eval_res["chunk_score"], update_best=is_updated)

if __name__ == "__main__":
    # Prepare model and dataset
	args = {
		"model": "RAGVT5",
		"dataset": "DUDE",
		"embed_model": "BGE",
		"reranker_model": "BGE",
		"page_retrieval": "Concat",
		"add_sep_token": False,
		"use_not_answerable_classifier": "MLP", # MLP / None
		"batch_size": 20,
		"batch_size_eval": 130,
		"chunk_num": 20,
		"chunk_size": 60,
		"chunk_size_tol": 0.2,
		"overlap": 10,
		"include_surroundings": 0,
		"model_weights": "/data/users/elopez/checkpoints/ragvt5_concat_dude_train_generator_dude/best.ckpt",
		"embed_weights": "/data/users/elopez/models/bge-finetuned/checkpoint-820",
		"reranker_weights": "BAAI/bge-reranker-v2-m3",
		"reorder_chunks": False,
		"rerank_filter_tresh": 0,
		"rerank_max_chunk_num": 10,
		"rerank_min_chunk_num": 1,
		"train_embedder": False, # Not implemented
		"train_language_backbone": False,
		"train_spatial_embedding": False,
		"train_visual_embedding": False,

		"train_not_answerable_classifier": True,
		"balance_nac_dataset": True,
		"nac_dataset_ratio": 0.5,
		"nac_loss_weight": 1.0,
		"not_answerable_mlp": {
			"hidden_dim": 256,
			"num_layers": 2,
			"emb_dim": 768
		}
	}
	# args = {
	# 	"use_RAG": True,
	# 	"model": "RAGPix2Struct",
	# 	"layout_model": "DIT",
	# 	"dataset": "Infographics", # MP-DocVQA / Infographics / DUDE / SP-DocVQA
	# 	"batch_size": 4,
	# 	"batch_size_eval": 8,
	# 	"layout_batch_size": 4,
	# 	"embedder_batch_size": 16,
	# 	"use_layout_labels": True,
	# 	"chunk_mode": "horizontal",
	# 	"chunk_num": 5,
	# 	"include_surroundings": (0,0),
	# 	"model_weights": "google/pix2struct-docvqa-base",
	# 	"layout_model_weights": "cmarkea/dit-base-layout-detection",
	# 	"use_precomputed_layouts": True,
	# 	"precomputed_layouts_path": "/data/users/elopez/infographics/images_layouts_dit_s2_spa.npz",
	# 	"cluster_layouts": True,
	# 	"cluster_mode": "spatial",
	# 	"calculate_n_clusters": "best"
	# }
	extra_args = {
		# "visible_devices": "0,1,2,3,4",
		"device": "cuda:0",
		"save_folder": "33-not-answerable-experiment",
		"save_name_append": "dude-mlp-train",
		"eval_start": False,
		"train_size": 1.0,
		"val_size": 1.0,
		"log_wandb": True,
		"log_media_interval": 10,
		"return_scores_by_sample": False,
		"return_answers": False,
		"save_results": False,
		"save_continuously": True,
		"compute_stats": False,
		"compute_stats_examples": False,
		"n_stats_examples": 0,
		"print_first_samples": -1
	}
	args.update(extra_args)

	# os.environ["CUDA_VISIBLE_DEVICES"] = args["visible_devices"]
	args = argparse.Namespace(**args)
	config = load_config(args)
	start_time = time.time()
	experiment_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	experiment_name = f"{config['model_name']}_{config['page_retrieval']}_{config['save_name_append']}_{experiment_date}"
	filename = f"{experiment_name}.json"
	print(f"Metrics will be saved in {config['save_dir']}/metrics/{config['save_folder']}/{filename}")

	print("Building model...")
	config["page_retrieval"] = config["page_retrieval"].lower()
	model = build_model(config)
	model.to(config["device"])
	print("Building dataset...")
	train_dataset = build_dataset(config, split="train", size=config["train_size"])
	val_dataset   = build_dataset(config, split="val", size=config["val_size"])
	train_data_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=mpdocvqa_collate_fn)
	val_data_loader   = DataLoader(val_dataset, batch_size=config["batch_size_eval"], shuffle=False, collate_fn=mpdocvqa_collate_fn)

	print("Training...")
	evaluator = Evaluator(config, case_sensitive=False)
	logger_train = Logger(config, experiment_name)
	log_media_interval = (len(val_data_loader) // config["log_media_interval"]) if config["save_continuously"] else 1
	logger_eval = LoggerEval(config, experiment_name, log_media_interval)
	train(
		train_data_loader, val_data_loader,
		model, evaluator, logger_train, logger_eval,
		config,
		filename=filename,
		start_time=start_time
	)
