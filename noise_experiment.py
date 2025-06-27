import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,4"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import yaml
import time
import datetime
import argparse
import numpy as np
import gc
import torch
from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader
from src.MP_DocVQA import mpdocvqa_collate_fn
from src.metrics import Evaluator
from src.utils import time_stamp_to_hhmmss, load_config, save_json
from src.build_utils import build_model, build_dataset
from src.RAGVT5 import RAGVT5
from src.HiVT5 import Proxy_HiVT5
from src.MMLongBenchDoc import MMLongBenchDoc
from src.logger import LoggerEval
from typing import Union


def evaluate(model, data_loader, evaluator, config):
	total_accuracies = []
	total_anls = []
	total_ret_prec = []
	total_chunk_scores = []
	total_seed_pages = []
	n_samples = 0
	out_of_memory = 0

	model.eval()

	for b, batch in enumerate(tqdm(data_loader)):
		bs = len(batch["question_id"])
		n_samples += bs

		try:
			_, pred_answers, _, pred_answers_conf, retrieval = model.inference(
					batch,
					return_retrieval=True
				)
			pred_answer_pages = retrieval.get("page_indices")
		except torch.OutOfMemoryError:
			print("Out of memory warning. Skipping batch.")
			pred_answers = None
			pred_answer_pages = None
			pred_answers_conf = None
			retrieval = {}
			out_of_memory += 1
			gc.collect()
			torch.cuda.empty_cache()
		
		metrics = evaluator.get_metrics(batch["answers"], pred_answers, batch.get("answer_type"), retrieval.get("top_k_layout_labels"))
		
		# Evaluate retrieval
		if "answer_page_idx" in batch and pred_answer_pages is not None:
			ret_metric = evaluator.get_retrieval_metric(batch["answer_page_idx"], pred_answer_pages)
		else:
			ret_metric = [0 for _ in range(bs)]
		if config["model_name"] == "RAGVT5":
			ret_eval = evaluator.eval_retrieval(batch, retrieval)
		else:
			ret_eval = {
				"chunk_score": [0 for _ in range(bs)]
			}
		
		total_accuracies.extend(metrics.get("accuracy", [0]*bs))
		total_anls.extend(metrics.get("anls", [0]*bs))
		total_ret_prec.extend(ret_metric)
		total_chunk_scores.extend(ret_eval["chunk_score"])
		total_seed_pages.extend(batch["num_seed_pages"])

		# Free memory
		del pred_answers, pred_answer_pages, pred_answers_conf, metrics, ret_metric, ret_eval, batch
		gc.collect()
		torch.cuda.empty_cache()
	
	to_return = {
		"total_accuracies": total_accuracies,
		"total_anls": total_anls,
		"total_ret_prec": total_ret_prec,
		"total_chunk_scores": total_chunk_scores,
		"total_seed_pages": total_seed_pages,
		"n_samples": n_samples,
		"out_of_memory": out_of_memory
	}

	return to_return


def run_experiment(model, config, noise_pages, repetitions):
	experiment_config = config.copy()
	experiment_config["noise_pages"] = noise_pages
	results = []

	print(f"Evaluating with {noise_pages} noise pages, {repetitions} repetitions...")
	evaluator = Evaluator(experiment_config, case_sensitive=False)
	
	for i in range(repetitions):
		experiment_config["noise_seed"] = 42+i
		val_dataset = build_dataset(experiment_config, split="val", size=config["val_size"])
		val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], collate_fn=mpdocvqa_collate_fn)
		results_i = evaluate(model, val_dataloader, evaluator, experiment_config)
		results.append(results_i)

	# Average per sample
	total_accuracies = np.mean([r["total_accuracies"] for r in results], axis=0)
	total_anls = np.mean([r["total_anls"] for r in results], axis=0)
	total_ret_prec = np.mean([r["total_ret_prec"] for r in results], axis=0)
	total_chunk_scores = np.mean([r["total_chunk_scores"] for r in results], axis=0)
	total_seed_pages = np.mean([r["total_seed_pages"] for r in results], axis=0)
	out_of_memory = results[0]["out_of_memory"]
	total_accuracies_std = np.std([r["total_accuracies"] for r in results], axis=0)
	total_anls_std = np.std([r["total_anls"] for r in results], axis=0)
	total_ret_prec_std = np.std([r["total_ret_prec"] for r in results], axis=0)
	total_chunk_scores_std = np.std([r["total_chunk_scores"] for r in results], axis=0)

	# Distribute per pages
	accuracy_per_page = np.zeros(int(total_seed_pages.max()))
	anls_per_page = np.zeros(int(total_seed_pages.max()))
	ret_prec_per_page = np.zeros(int(total_seed_pages.max()))
	chunk_scores_per_page = np.zeros(int(total_seed_pages.max()))
	accuracy_per_page_std = np.zeros(int(total_seed_pages.max()))
	anls_per_page_std = np.zeros(int(total_seed_pages.max()))
	ret_prec_per_page_std = np.zeros(int(total_seed_pages.max()))
	chunk_scores_per_page_std = np.zeros(int(total_seed_pages.max()))
	samples_per_page = np.zeros(int(total_seed_pages.max()))

	# Convert total_seed_pages to integers for indexing
	total_seed_pages_int = total_seed_pages.astype(int)

	for i in range(len(total_accuracies)):
		page_idx = total_seed_pages_int[i]-1
		accuracy_per_page[page_idx] += total_accuracies[i]
		anls_per_page[page_idx] += total_anls[i]
		ret_prec_per_page[page_idx] += total_ret_prec[i]
		chunk_scores_per_page[page_idx] += total_chunk_scores[i]
		accuracy_per_page_std[page_idx] += total_accuracies_std[i]
		anls_per_page_std[page_idx] += total_anls_std[i]
		ret_prec_per_page_std[page_idx] += total_ret_prec_std[i]
		chunk_scores_per_page_std[page_idx] += total_chunk_scores_std[i]
		samples_per_page[page_idx] += 1

	# Normalize by samples per page
	for page_idx in range(len(samples_per_page)):
		if samples_per_page[page_idx] > 0:
			accuracy_per_page[page_idx] /= samples_per_page[page_idx]
			anls_per_page[page_idx] /= samples_per_page[page_idx]
			ret_prec_per_page[page_idx] /= samples_per_page[page_idx]
			chunk_scores_per_page[page_idx] /= samples_per_page[page_idx]
			accuracy_per_page_std[page_idx] /= samples_per_page[page_idx]
			anls_per_page_std[page_idx] /= samples_per_page[page_idx]
			ret_prec_per_page_std[page_idx] /= samples_per_page[page_idx]
			chunk_scores_per_page_std[page_idx] /= samples_per_page[page_idx]

	print("Out of memory for this experiment:", out_of_memory)
	to_return = {
		"samples_per_page": list(samples_per_page),
		"accuracy_per_page": list(accuracy_per_page),
		"anls_per_page": list(anls_per_page),
		"ret_prec_per_page": list(ret_prec_per_page),
		"chunk_scores_per_page": list(chunk_scores_per_page),
		"accuracy_per_page_std": list(accuracy_per_page_std),
		"anls_per_page_std": list(anls_per_page_std),
		"ret_prec_per_page_std": list(ret_prec_per_page_std),
		"chunk_scores_per_page_std": list(chunk_scores_per_page_std)
	}
	print(to_return)

	return to_return


if __name__ == "__main__":
	# Prepare model and dataset
	args = {
		"use_RAG": True,
		"model": "RAGVT5",
		"dataset": "DUDE-Noise", # MP-DocVQA / Infographics / DUDE / MMLongBenchDoc
		"embed_model": "BGE", # BGE / VT5 / JINA
		"reranker_model": "BGE",
		"page_retrieval": "Concat", # Oracle / Concat / Logits / Maxconf / AnyConf / MaxConfPage / AnyConfPage / MajorPage / WeightMajorPage / AnyConfOracle / Custom (HiVT5 only)
		"add_sep_token": False,
		"batch_size": 1, # 50 Oracle / Concat / MajorPage / WeightMajorPage / AnyConfOracle, 32 MaxConf / AnyConf, 16 MaxConfPage / AnyConfPage
		"chunk_num": 20,
		"chunk_size": 60,
		"chunk_size_tol": 0.2,
		"overlap": 10,
		"include_surroundings": 0,
		# "model_weights": "Qwen/Qwen2.5-VL-7B-Instruct",
		# "model_weights": "/data/users/elopez/checkpoints/ragvt5_concat_mp-docvqa_train_generator_mpdocvqa/best.ckpt",
		"model_weights": "/data/users/elopez/checkpoints/ragvt5_concat_dude_train_generator_dude/best.ckpt",
		# "model_weights": "rubentito/vt5-base-spdocvqa",
		# "embed_weights": "BAAI/bge-small-en-v1.5",
		"embed_weights": "/data/users/elopez/models/bge-finetuned/checkpoint-820", # or VT5
		# "embed_weights": "/data/users/elopez/models/bge-finetuned-info-30/checkpoint-540",
		"reranker_weights": "BAAI/bge-reranker-v2-m3",
		"lora_weights": "",
		# "lora_weights": "/data/users/elopez/checkpoints/RAGVT5_lora_2025-03-31_09-52-23/checkpoint-900",
		"reorder_chunks": False,
		"rerank_filter_tresh": 0,
		"rerank_max_chunk_num": 10,
		"rerank_min_chunk_num": 1,
		# "use_precomputed_layouts": True,
		# "precomputed_layouts_path": "/data/users/elopez/data/images_layouts_dit_s2_spa_sem.npz"
	}

	extra_args = {
		# "visible_devices": "2,4",
		"device": "cuda:0",
		"save_folder": "32-experiment-noise",
		"save_name_append": "noise-pages-v2-ragvt5-dude",
		"val_size": 1.0,
		"log_wandb": False,
		"log_media_interval": 10,
		"return_scores_by_sample": True,
		"return_answers": True,
		"save_results": True,
		"save_continuously": True,
		"compute_stats": False,
		"compute_stats_examples": False,
		"n_stats_examples": 5,
	}
	args.update(extra_args)
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
	evaluator = Evaluator(config, case_sensitive=False)

	results = {}

	for noise_pages in [100, 20, 3, 0]:
		print("-"*50)
		print(f"Evaluating with {noise_pages} noise pages...")
		results[noise_pages] = run_experiment(model, config, noise_pages, 1)

	save_dict = {
		"config": config,
		"results_by_noise_pages": results
	}

	save_json(f"{config['save_dir']}/metrics/{config['save_folder']}/{experiment_name}.json", save_dict)
