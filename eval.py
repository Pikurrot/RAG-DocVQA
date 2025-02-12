import os
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
from src.logger import LoggerEval
from typing import Union


def save_local(
		config: dict,
		filename: str,
		eval_res: dict
):
	inf_time_f = eval_res["inf_time"]
	inf_time = time_stamp_to_hhmmss(inf_time_f, string=True)
	avg_load_time = time_stamp_to_hhmmss(eval_res["avg_load_time"], string=True)
	avg_layout_time = time_stamp_to_hhmmss(eval_res["avg_layout_time"], string=True)
	avg_retrieval_time = time_stamp_to_hhmmss(eval_res["avg_retrieval_time"], string=True)
	avg_generation_time = time_stamp_to_hhmmss(eval_res["avg_generation_time"], string=True)
	total_load_time = time_stamp_to_hhmmss(eval_res["total_load_time"], string=True)
	total_layout_time = time_stamp_to_hhmmss(eval_res["total_layout_time"], string=True)
	total_retrieval_time = time_stamp_to_hhmmss(eval_res["total_retrieval_time"], string=True)
	total_generation_time = time_stamp_to_hhmmss(eval_res["total_generation_time"], string=True)

	for key, stat in eval_res["retrieval_stats"].items():
		if isinstance(stat, Counter):
			if len(list(stat.elements())) == 0:
				eval_res["retrieval_stats"][key] = {}
				continue
			mean = np.mean(list(stat.elements()))
			std = np.std(list(stat.elements()))
			min_val = min(stat.elements())
			max_val = max(stat.elements())
			# save only relevant distribution
			stat_relevant = dict()
			# top 5 most common values
			count = {k: v for k, v in stat.most_common(5)}
			examples = {k: eval_res["retrieval_stats_examples"][key][k] for k in count.keys()}
			stat_relevant["most_common"] = {k: {"count": count[k], "examples": examples[k]} for k in count}
			# top 5 least common values
			count = {k: v for k, v in stat.most_common()[:-6:-1]}
			examples = {k: eval_res["retrieval_stats_examples"][key][k] for k in count.keys()}
			stat_relevant["least_common"] = {k: {"count": count[k], "examples": examples[k]} for k in count}
			# smallest 5 keys
			count = {k: stat[k] for k in sorted(stat.keys())[:5]}
			examples = {k: eval_res["retrieval_stats_examples"][key][k] for k in count.keys()}
			stat_relevant["smallest"] = {k: {"count": count[k], "examples": examples[k]} for k in count}
			# largest 5 keys
			count = {k: stat[k] for k in sorted(stat.keys())[:-6:-1]}
			examples = {k: eval_res["retrieval_stats_examples"][key][k] for k in count.keys()}
			stat_relevant["largest"] = {k: {"count": count[k], "examples": examples[k]} for k in count}
			# top 5 keys around the mean
			count = {k: stat[k] for k in sorted(stat.keys(), key=lambda x: abs(x-mean))[:5]}
			examples = {k: eval_res["retrieval_stats_examples"][key][k] for k in count.keys()}
			stat_relevant["around_mean"] = {k: {"count": count[k], "examples": examples[k]} for k in count}

			eval_res["retrieval_stats"][key] = {
				"mean": mean,
				"std": std,
				"min": min_val,
				"max": max_val,
				"relevant_samples": stat_relevant
			}
		elif isinstance(stat, dict) and isinstance(list(stat.values())[0], list):
			eval_res["retrieval_stats"][key] = {k: np.mean(v) if v else -1 for k, v in stat.items()}
	
	save_data = {
		"Model": config["model_name"],
		"Model weights": config["model_weights"],
		"Embed weights": config.get("embed_weights", "-"),
		"Layout model weigths": config.get("layout_model_weights", "-"),
		"Dataset": config["dataset_name"],
		"Page retrieval": config.get("page_retrieval", "-").capitalize(),
		"embed_model": config.get("embed_model", "-"),
		"chunk_num": config.get("chunk_num", "-"),
		"chunk_size": config.get("chunk_size", "-"),
		"overlap": config.get("overlap", "-"),
		"include_surroundings": config.get("include_surroundings", "-"),
		"Avg accuracy": eval_res["accuracy"],
		"Avg ANLS": eval_res["anls"],
		"Avg retrieval precision": eval_res["retrieval_precision"],
		"Avg chunk score": eval_res["chunk_score"],
		"Dataset size": f"{config['val_size']*100}%",
		"Inference time": inf_time,
		"Avg load time": avg_load_time,
		"Avg retrieval time (layout)": avg_layout_time,
		"Avg retrieval time": avg_retrieval_time,
		"Avg generation time": avg_generation_time,
		"Total load time": f"{total_load_time} ({eval_res['total_load_time']/inf_time_f*100:.2f}%)",
		"Total retrieval time (layout)": f"{total_layout_time} ({eval_res['total_layout_time']/inf_time_f*100:.2f}%)",
		"Total retrieval time": f"{total_retrieval_time} ({eval_res['total_retrieval_time']/inf_time_f*100:.2f}%)",
		"Total generation time": f"{total_generation_time} ({eval_res['total_generation_time']/inf_time_f*100:.2f}%)",
		"Retrieval stats": eval_res["retrieval_stats"],
		"Scores by samples": eval_res["scores_by_samples"]
	}

	metrics_file = os.path.join(config["save_dir"], "metrics", config["save_folder"], filename)
	results_file = os.path.join(config["save_dir"], "results", config["save_folder"], filename)
	save_json(metrics_file, save_data, smart=True, smart_start_level=5)
	if eval_res["results"]:
		save_json(results_file, eval_res["results"])

	return save_data

def log_wandb(
		logger: LoggerEval,
		save_data: dict,
		config: dict
):
	def str2sec(time_str: str) -> int:
		# takes something like "00:00:01" or "00:00:01 (2.90%)" and returns the seconds
		if "(" in time_str:
			time_str = time_str.split("(")[0].strip()
		# transform date to total seconds
		h, m, s = time_str.split(":")
		return int(h)*3600 + int(m)*60 + int(s)

	log_data = {
		"Val/Accuracy": save_data["Avg accuracy"],
		"Val/Anls": save_data["Avg ANLS"],
		"Val/Retrieval precision": save_data["Avg retrieval precision"],
		"Val/Chunk score": save_data["Avg chunk score"],
		"Val/Avg. inference times": {
			"values": {
				"Load time": str2sec(save_data["Avg load time"]),
				"Retrieval time": str2sec(save_data["Avg retrieval time"]),
				"Generation time": str2sec(save_data["Avg generation time"])
			},
			"config": {
				"chart_type": "pie"
			}
		},
		"Val/Avg. retrieval times": {
				"values": {
					"Layout time": str2sec(save_data["Avg retrieval time (layout)"]),
					"Rest": str2sec(save_data["Avg retrieval time"]) - str2sec(save_data["Avg retrieval time (layout)"]),
				},
				"config": {
					"chart_type": "pie"
				}
			}
	}
	if config["layout_model_weights"] and config["page_retrieval"] != "oracle" and config["compute_stats"]:
		log_data.update({
			"Val/Layout labels count": {
				"values": [
					save_data["Retrieval stats"]["layout_labels_dist"],
					save_data["Retrieval stats"]["layout_labels_topk_dist"]
				],
				"config": {
					"chart_type": "spider",
					"log_scale": True,
					"legend": ["All labels", "Top-k chunks"]
				}
			},
			"Val/Layout labels metrics": {
				"values": [
					save_data["Retrieval stats"]["layout_labels_accuracy"],
					save_data["Retrieval stats"]["layout_labels_anls"]
				],
				"config": {
					"chart_type": "spider",
					"legend": ["Accuracy", "ANLS"]
				}
			}
		})

	logger.parse_and_log(log_data)

def evaluate(
		data_loader: DataLoader,
		model: Union[RAGVT5, Proxy_HiVT5],
		evaluator: Evaluator,
		logger: LoggerEval,
		config: dict,
		**kwargs
):
	return_scores_by_sample = config["return_scores_by_sample"]
	return_answers = config["return_answers"]
	save_results = config["save_results"]
	save_continuously = config["save_continuously"]
	start_time = kwargs.get("start_time", time.time())
	filename = kwargs.get("filename", "eval.json")
	model_name = model.__class__.__name__
	model_name = "Hi-VT5" if model_name == "Proxy_HiVT5" else model_name

	if return_scores_by_sample:
		scores_by_samples = {}
		total_accuracies = []
		total_anls = []
		total_ret_prec = []
		total_chunk_scores = []
	else:
		total_accuracies = 0
		total_anls = 0
		total_ret_prec = 0
		total_chunk_scores = 0
	load_time = 0
	layout_time = 0
	retrieval_time = 0
	generation_time = 0
	results = []
	retrieval_stats = {}
	retrieval_stats_examples = {}

	all_pred_answers = []
	model.eval()

	# Evaluate each batch
	for b, batch in enumerate(tqdm(data_loader)):
		bs = len(batch["question_id"])

		# Inference using the model
		if model_name == "Hi-VT5":
			start_gen_time = time.time()
			_, pred_answers, pred_answer_pages, pred_answers_conf = model.inference(
				batch,
				return_pred_answer=True
			)
			retrieval = {
				"retrieval_time": 0,
				"generation_time": time.time() - start_gen_time
			}
		elif model_name == "RAGVT5":
			_, pred_answers, _, pred_answers_conf, retrieval = model.inference(
				batch,
				return_retrieval=True
			)
			pred_answer_pages = retrieval["page_indices"]

		# Compute metrics
		metrics = evaluator.get_metrics(batch["answers"], pred_answers, batch.get("answer_type", None), retrieval["top_k_layout_labels"])

		# Evaluate retrieval
		if "answer_page_idx" in batch and pred_answer_pages is not None:
			ret_metric = evaluator.get_retrieval_metric(batch["answer_page_idx"], pred_answer_pages)
		else:
			ret_metric = [0 for _ in range(bs)]
		if model_name == "Hi-VT5":
			ret_eval = {
				"chunk_score": [0 for _ in range(bs)]
			}
		elif model_name == "RAGVT5":
			ret_eval = evaluator.eval_retrieval(batch, retrieval)

		if return_scores_by_sample:
			# Save metrics for each sample
			for _b in range(bs):
				scores_by_samples[batch["question_id"][_b]] = {
					"accuracy": metrics["accuracy"][_b],
					"anls": metrics["anls"][_b],
					"ret_prec": ret_metric[_b],
					"pred_answer": pred_answers[_b],
					"actual_answer": batch["answers"][_b],
					"pred_answer_conf": pred_answers_conf[_b],
					"pred_answer_page": pred_answer_pages[_b] if pred_answer_pages is not None else None,
					"chunk_score": ret_eval["chunk_score"][_b],
				}

		# Accumulate metrics for the whole dataset
		if return_scores_by_sample:
			total_accuracies.extend(metrics["accuracy"])
			total_anls.extend(metrics["anls"])
			total_ret_prec.extend(ret_metric)
			total_chunk_scores.extend(ret_eval["chunk_score"])
		else:
			total_accuracies += sum(metrics["accuracy"])
			total_anls += sum(metrics["anls"])
			total_ret_prec += sum(ret_metric)
			total_chunk_scores += sum(ret_eval["chunk_score"])
		load_time += sum(batch["load_time"])
		layout_time += retrieval["stats"]["layout_time"]	
		retrieval_time += retrieval["retrieval_time"]
		generation_time += retrieval["generation_time"]
		
		retrieval["stats"]["layout_labels_accuracy"] = metrics["layout_labels_accuracy"]
		retrieval["stats"]["layout_labels_anls"] = metrics["layout_labels_anls"]

		if return_answers:
			all_pred_answers.extend(pred_answers)

		if save_results:
			for i in range(bs):
				if isinstance(pred_answer_pages, int):
					answer_page = pred_answer_pages
				elif pred_answer_pages is None or len(pred_answer_pages) == 0 or len(pred_answer_pages[i]) == 0:
					answer_page = 0
				elif isinstance(pred_answer_pages[i], int):
					answer_page = pred_answer_pages[i]
				else:
					answer_page = pred_answer_pages[i][0]
				results.append({
					"questionId": batch["question_id"][i],
					"answer": pred_answers[i],
					"answer_page": answer_page
				})

		# Accumulate retrieval stats
		# Debug print retrieval stats sizes after merging each batch:
		# for key, stat in retrieval_stats.items():
		# 	if isinstance(stat, Counter):
		# 		counter_len = sum(stat.values())
		# 		print(f"Batch {b}: Counter '{key}' total count: {counter_len}")
		# 	elif isinstance(stat, dict):
		# 		dict_len = len(stat)
		# 		print(f"Batch {b}: Dict '{key}' length: {dict_len}")
				
		# # And similarly for retrieval_stats_examples:
		# for key, examples in retrieval_stats_examples.items():
		# 	if isinstance(examples, dict) and examples:
		# 		max_key = max(examples, key=lambda k: len(examples[k]))
		# 		max_len = len(examples[max_key])
		# 		print(f"Batch {b}: Stat examples key '{key}', max length key: '{max_key}' with {max_len} examples")

		del retrieval["stats"]["layout_time"]
		if b == 0:
			retrieval_stats = retrieval["stats"]
			retrieval_stats_examples = retrieval["stats_examples"]
		else:
			for key in retrieval_stats:
				if isinstance(retrieval_stats[key], Counter):
					retrieval_stats[key] += retrieval["stats"][key]
					for k in retrieval["stats_examples"][key]:
						if k not in retrieval_stats_examples[key]:
							retrieval_stats_examples[key][k] = []
						retrieval_stats_examples[key][k].extend(retrieval["stats_examples"][key][k])
						retrieval_stats_examples[key][k] = retrieval_stats_examples[key][k][:model.n_stats_examples]
				else:
					for k in retrieval_stats[key]:
						if isinstance(retrieval_stats[key][k], list):
							retrieval_stats[key][k].extend(retrieval["stats"][key][k])
						else:
							retrieval_stats[key][k] += retrieval["stats"][key][k]
		
		# Free memory
		del pred_answers, pred_answer_pages, pred_answers_conf, metrics, ret_metric, ret_eval, batch
		gc.collect()
		torch.cuda.empty_cache()
		# print(torch.cuda.memory_summary(device=None, abbreviated=False))

		# Save data
		if save_continuously or (b == len(data_loader) - 1):
			if return_scores_by_sample:
				accuracy = np.mean(total_accuracies)
				anls = np.mean(total_anls)
				retrieval_precision = np.mean(total_ret_prec)
				avg_chunk_score = np.mean(total_chunk_scores)
			else:
				# Compute average metrics
				accuracy = np.mean(total_accuracies/len(data_loader.dataset))
				anls = np.mean(total_anls/len(data_loader.dataset))
				retrieval_precision = np.mean(total_ret_prec/len(data_loader.dataset))
				avg_chunk_score = np.mean(total_chunk_scores/len(data_loader.dataset))
				scores_by_samples = []
			avg_load_time = load_time/len(data_loader)
			avg_layout_time = layout_time/len(data_loader)
			avg_retrieval_time = retrieval_time/len(data_loader)
			avg_generation_time = generation_time/len(data_loader)

			res = {
				"accuracy": accuracy,
				"anls": anls,
				"retrieval_precision": retrieval_precision,
				"chunk_score": avg_chunk_score,
				"all_pred_answers": all_pred_answers,
				"scores_by_samples": scores_by_samples,
				"inf_time": time.time() - start_time,
				"avg_load_time": avg_load_time,
				"avg_layout_time": avg_layout_time,
				"avg_retrieval_time": avg_retrieval_time,
				"avg_generation_time": avg_generation_time,
				"total_load_time": load_time,
				"total_layout_time": layout_time,
				"total_retrieval_time": retrieval_time,
				"total_generation_time": generation_time,
				"retrieval_stats": retrieval_stats.copy(),
				"retrieval_stats_examples": retrieval_stats_examples.copy(),
				"results": results
			}

			# Save data
			save_data = save_local(config, filename, res)

			# Log data
			log_wandb(logger, save_data, config)

	return res


if __name__ == "__main__":
	# Prepare model and dataset
	args = {
		"model": "RAGVT5",
		"dataset": "MP-DocVQA",
		"embed_model": "BGE", # BGE, VT5, BGE-M3, BGE-reranker
		"page_retrieval": "Concat", # Oracle / Concat / Logits / Maxconf / AnyConf / MaxConfPage / AnyConfPage / MajorPage / WeightMajorPage / AnyConfOracle / Custom (HiVT5 only)
		"add_sep_token": False,
		"batch_size": 50, # 50 Oracle / Concat / MajorPage / WeightMajorPage / AnyConfOracle, 32 MaxConf / AnyConf, 16 MaxConfPage / AnyConfPage
		"layout_batch_size": 8,
		"chunk_num": 10,
		"chunk_size": 60,
		"chunk_size_tol": 0.2,
		"overlap": 10,
		"include_surroundings": 0,
		# "model_weights": "/data3fast/users/elopez/checkpoints/ragvt5_concat_mp-docvqa_sep-token/model__9.ckpt"
		"embed_weights": "/data3fast/users/elopez/models/bge-finetuned-2/checkpoint-820",
		"layout_model_weights": "cmarkea/dit-base-layout-detection",
		"use_layout_labels": False,
		"use_precomputed_layouts": True,
	}
	extra_args = {
		"visible_devices": "5",
		"save_folder": "9-train_generator_with_layout",
		"save_name_append": "untraiend",
		"val_size": 1.0,
		"log_wandb": True,
		"log_media_interval": 10,
		"return_scores_by_sample": True,
		"return_answers": True,
		"save_results": False,
		"save_continuously": True,
		"compute_stats": False,
		"compute_stats_examples": True,
		"n_stats_examples": 5,
	}
	args.update(extra_args)

	# Override args with yaml file if provided
	if len(sys.argv) > 1 and sys.argv[1].endswith('.yml'):
		yaml_path = sys.argv[1]
		with open(yaml_path, 'r') as f:
			yaml_data = yaml.safe_load(f)
		# Flatten all top-level sub-dicts into a single dict
		flattened_yaml = {}
		for _, subdict in yaml_data.items():
			if isinstance(subdict, dict):
				flattened_yaml.update(subdict)
		args.update(flattened_yaml)

	# Parse additional CLI "key=value" overrides
	for arg in sys.argv[2:]:
		if '=' in arg:
			k, v = arg.split('=', 1)
			if v.lower() in ['true', 'false']:
				v = (v.lower() == 'true')
			elif v.isdigit():
				v = int(v)
			args[k] = v

	if isinstance(args["visible_devices"], list):
		os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in args["visible_devices"])
	else:
		os.environ["CUDA_VISIBLE_DEVICES"] = str(args["visible_devices"])

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
	dataset = build_dataset(config, split="val", size=config["val_size"], use_precomputed_layouts=config["use_precomputed_layouts"])
	val_data_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=mpdocvqa_collate_fn, num_workers=0)

	# Evaluate the model
	print("Evaluating...")
	evaluator = Evaluator(case_sensitive=False)
	log_media_interval = (len(val_data_loader) // config["log_media_interval"]) if config["save_continuously"] else 1
	logger = LoggerEval(config, experiment_name, log_media_interval)
	evaluate(
		val_data_loader,
		model, evaluator, logger,
		config,
		filename=filename,
		start_time=start_time
	)
