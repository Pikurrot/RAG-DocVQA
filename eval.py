import os
import time
import datetime
import argparse
import numpy as np
import gc
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.MP_DocVQA import mpdocvqa_collate_fn
from src.metrics import Evaluator
from src.utils import time_stamp_to_hhmmss, load_config, save_json
from src.build_utils import build_model, build_dataset
from src.RAG_VT5 import RAGVT5
from src.HiVT5 import Proxy_HiVT5
from typing import Union


def evaluate(
		data_loader: DataLoader,
		model: Union[RAGVT5, Proxy_HiVT5],
		evaluator: Evaluator,
		**kwargs
):
	return_scores_by_sample = kwargs.get("return_scores_by_sample", False)
	return_answers = kwargs.get("return_answers", False)
	save_results = kwargs.get("save_results", False)
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
	retrieval_time = 0
	generation_time = 0
	results = []

	all_pred_answers = []
	model.eval()

	# Evaluate each batch
	for b, batch in enumerate(tqdm(data_loader)):
		bs = len(batch["question_id"])

		# Inference using the model
		if model_name == "Hi-VT5":
			start_time = time.time()
			_, pred_answers, pred_answer_pages, pred_answers_conf = model.inference(
				batch,
				return_pred_answer=True
			)
			retrieval = {
				"retrieval_time": 0,
				"generation_time": time.time() - start_time
			}
		elif model_name == "RAGVT5":
			_, pred_answers, _, pred_answers_conf, retrieval = model.inference(
				batch,
				return_retrieval=True,
				chunk_num=kwargs.get("chunk_num", 5),
				chunk_size=kwargs.get("chunk_size", 30),
				overlap=kwargs.get("overlap", 0),
				include_surroundings=kwargs.get("include_surroundings", 10)
			)
			pred_answer_pages = retrieval["page_indices"]

		# Compute metrics
		metrics = evaluator.get_metrics(batch["answers"], pred_answers, batch.get("answer_type", None))

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
			for b in range(bs):
				scores_by_samples[batch["question_id"][b]] = {
					"accuracy": metrics["accuracy"][b],
					"anls": metrics["anls"][b],
					"ret_prec": ret_metric[b],
					"pred_answer": pred_answers[b],
					"actual_answer": batch["answers"][b],
					"pred_answer_conf": pred_answers_conf[b],
					"pred_answer_page": pred_answer_pages[b] if pred_answer_pages is not None else None,
					"chunk_score": ret_eval["chunk_score"][b],
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
		retrieval_time += retrieval["retrieval_time"]
		generation_time += retrieval["generation_time"]

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
		
		# Free memory
		del pred_answers, pred_answer_pages, pred_answers_conf, metrics, ret_metric, ret_eval, batch
		gc.collect()
		torch.cuda.empty_cache()
		# print(torch.cuda.memory_summary(device=None, abbreviated=False))

	if not return_scores_by_sample:
		# Compute average metrics
		total_accuracies = total_accuracies/len(data_loader.dataset)
		total_anls = total_anls/len(data_loader.dataset)
		total_ret_prec = total_ret_prec/len(data_loader.dataset)
		total_chunk_scores = total_chunk_scores/len(data_loader.dataset)
		scores_by_samples = []
	avg_load_time = load_time/len(data_loader)
	avg_retrieval_time = retrieval_time/len(data_loader)
	avg_generation_time = generation_time/len(data_loader)

	return {
		"accuracy": total_accuracies,
		"anls": total_anls,
		"retrieval_precision": total_ret_prec,
		"chunk_score": total_chunk_scores,
		"all_pred_answers": all_pred_answers,
		"scores_by_samples": scores_by_samples,
		"avg_load_time": avg_load_time,
		"avg_retrieval_time": avg_retrieval_time,
		"avg_generation_time": avg_generation_time,
		"total_load_time": load_time,
		"total_retrieval_time": retrieval_time,
		"total_generation_time": generation_time,
		"results": results
	}


if __name__ == "__main__":
	# Prepare model and dataset
	args = {
		"model": "RAGVT5", # RAGVT5, HiVT5
		"dataset": "MP-DocVQA",
		"embed_model": "BGE", # BGE, VT5
		"page_retrieval": "AnyConfOracle", # Oracle / Concat / Logits / Maxconf / AnyConf / MaxConfPage / AnyConfPage / MajorPage / WeightMajorPage / AnyConfOracle / Custom (HiVT5 only)
		"add_sep_token": False,
		"batch_size": 32, # 50 Oracle / Concat / MajorPage / WeightMajorPage / AnyConfOracle, 32 MaxConf / AnyConf, 16 MaxConfPage / AnyConfPage
		"chunk_num": 10,
		"chunk_size": 60,
		"overlap": 10,
		"include_surroundings": 0,
		"visible_devices": "0",
		# "model_weights": "/data3fast/users/elopez/checkpoints/ragvt5_concat_mp-docvqa_sep-token/model__9.ckpt"
	}
	os.environ["CUDA_VISIBLE_DEVICES"] = args["visible_devices"]
	args = argparse.Namespace(**args)
	config = load_config(args)
	start_time = time.time()
	print("Building model...")
	model = build_model(config)
	model.to(config["device"])
	print("Building dataset...")
	data_size = 1.0
	dataset = build_dataset(config, split="val", size=data_size)
	val_data_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=mpdocvqa_collate_fn, num_workers=0)

	# Evaluate the model
	print("Evaluating...")
	evaluator = Evaluator(case_sensitive=False)
	eval_res = evaluate(
		val_data_loader,
		model, evaluator,
		return_scores_by_sample=True,
		return_answers=True,
		save_results=False,
		chunk_num=config.get("chunk_num", 10),
		chunk_size=config.get("chunk_size", 60),
		overlap=config.get("overlap", 10),
		include_surroundings=config.get("include_surroundings", 0)
	)
	accuracy = np.mean(eval_res["accuracy"])
	anls = np.mean(eval_res["anls"])
	retrieval_precision = np.mean(eval_res["retrieval_precision"])
	avg_chunk_score = np.mean(eval_res["chunk_score"])
	
	# Save results
	inf_time_f = time.time() - start_time
	inf_time = time_stamp_to_hhmmss(inf_time_f, string=True)
	avg_load_time = time_stamp_to_hhmmss(eval_res["avg_load_time"], string=True)
	avg_retrieval_time = time_stamp_to_hhmmss(eval_res["avg_retrieval_time"], string=True)
	avg_generation_time = time_stamp_to_hhmmss(eval_res["avg_generation_time"], string=True)
	total_load_time = time_stamp_to_hhmmss(eval_res["total_load_time"], string=True)
	total_retrieval_time = time_stamp_to_hhmmss(eval_res["total_retrieval_time"], string=True)
	total_generation_time = time_stamp_to_hhmmss(eval_res["total_generation_time"], string=True)
	save_data = {
		"Model": config["model_name"],
		"Model_weights": config["model_weights"],
		"Dataset": config["dataset_name"],
		"Page retrieval": config.get("page_retrieval", "-").capitalize(),
		"embed_model": config.get("embed_model", "-"),
		"chunk_num": config.get("chunk_num", "-"),
		"chunk_size": config.get("chunk_size", "-"),
		"overlap": config.get("overlap", "-"),
		"include_surroundings": config.get("include_surroundings", "-"),
		"Avg accuracy": accuracy,
		"Avg ANLS": anls,
		"Avg retrieval precision": retrieval_precision,
		"Avg chunk score": avg_chunk_score,
		"Dataset size": f"{data_size*100}%",
		"Inference time": inf_time,
		"Avg load time": avg_load_time,
		"Avg retrieval time": avg_retrieval_time,
		"Avg generation time": avg_generation_time,
		"Total load time": f"{total_load_time} ({eval_res['total_load_time']/inf_time_f*100:.2f}%)",
		"Total retrieval time": f"{total_retrieval_time} ({eval_res['total_retrieval_time']/inf_time_f*100:.2f}%)",
		"Total generation time": f"{total_generation_time} ({eval_res['total_generation_time']/inf_time_f*100:.2f}%)",
		"Scores by samples": eval_res["scores_by_samples"]
	}
	experiment_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	if config["model_name"] == "Hi-VT5":
		filename = "hivt5_{:}.json".format(experiment_date)
	else:
		filename = "{:}_{:}_{:}.json".format(config.get("embed_model", "").lower(), config.get("page_retrieval", "").lower(), experiment_date)
	metrics_file = os.path.join(config["save_dir"], "metrics", filename)
	results_file = os.path.join(config["save_dir"], "results", filename)
	save_json(metrics_file, save_data)
	print("Metrics correctly saved in: {:s}".format(metrics_file))
	if eval_res["results"]:
		save_json(results_file, eval_res["results"])
		print("Results correctly saved in: {:s}".format(results_file))
