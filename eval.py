import os
import time
import datetime
import argparse
import numpy as np
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.MP_DocVQA import mpdocvqa_collate_fn
from src.metrics import Evaluator
from src.utils import time_stamp_to_hhmmss, load_config, save_json
from src.build_utils import build_model, build_dataset
from src.RAG_VT5 import RAGVT5


def evaluate(
		data_loader: DataLoader,
		model: RAGVT5,
		evaluator: Evaluator,
		**kwargs
):
	return_scores_by_sample = kwargs.get("return_scores_by_sample", False)
	return_answers = kwargs.get("return_answers", False)

	if return_scores_by_sample:
		scores_by_samples = {}
		total_accuracies = []
		total_anls = []
		total_ret_prec = []
	else:
		total_accuracies = 0
		total_anls = 0
		total_ret_prec = 0
	load_time = 0
	retrieval_time = 0
	generation_time = 0

	all_pred_answers = []
	model.eval()

	# Evaluate each batch
	for b, batch in enumerate(tqdm(data_loader)):
		bs = len(batch["question_id"])

		# Inference using the model
		outputs, pred_answers, _, pred_answers_conf, retrieval = \
			model.inference(batch, return_retrieval=True, include_surroundings=10, k=5)
		pred_answer_pages = retrieval["page_indices"]

		# Compute metrics
		metrics = evaluator.get_metrics(batch["answers"], pred_answers, batch.get("answer_type", None))

		# Compute retrieval metric
		if "answer_page_idx" in batch and pred_answer_pages is not None:
			ret_metric = evaluator.get_retrieval_metric(batch["answer_page_idx"], pred_answer_pages)
		else:
			ret_metric = [0 for _ in range(bs)]

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
					"pred_answer_page": pred_answer_pages[b] if pred_answer_pages is not None else None
				}

		# Accumulate metrics for the whole dataset
		if return_scores_by_sample:
			total_accuracies.extend(metrics["accuracy"])
			total_anls.extend(metrics["anls"])
			total_ret_prec.extend(ret_metric)
		else:
			total_accuracies += sum(metrics["accuracy"])
			total_anls += sum(metrics["anls"])
			total_ret_prec += sum(ret_metric)
		load_time += sum(batch["load_time"])
		retrieval_time += retrieval["retrieval_time"]
		generation_time += retrieval["generation_time"]

		if return_answers:
			all_pred_answers.extend(pred_answers)
		
		gc.collect()

	if not return_scores_by_sample:
		# Compute average metrics
		total_accuracies = total_accuracies/len(data_loader.dataset)
		total_anls = total_anls/len(data_loader.dataset)
		total_ret_prec = total_ret_prec/len(data_loader.dataset)
		scores_by_samples = []
	avg_load_time = load_time/len(data_loader)
	avg_retrieval_time = retrieval_time/len(data_loader)
	avg_generation_time = generation_time/len(data_loader)

	return {
		"accuracy": total_accuracies,
		"anls": total_anls,
		"retrieval_precision": total_ret_prec,
		"all_pred_answers": all_pred_answers,
		"scores_by_samples": scores_by_samples,
		"avg_load_time": avg_load_time,
		"avg_retrieval_time": avg_retrieval_time,
		"avg_generation_time": avg_generation_time,
		"total_load_time": load_time,
		"total_retrieval_time": retrieval_time,
		"total_generation_time": generation_time,
	}


if __name__ == "__main__":
	# Prepare model and dataset
	args = {
		"model": "VT5",
		"dataset": "MP-DocVQA",
		"embed_model": "VT5" # VT5 or BGE
	}
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
	eval_res = evaluate(val_data_loader, model, evaluator, return_scores_by_sample=False, return_answers=True)
	accuracy = np.mean(eval_res["accuracy"])
	anls = np.mean(eval_res["anls"])
	answ_page_pred_acc = np.mean(eval_res["retrieval_precision"])
	
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
		"Avg accuracy": accuracy,
		"Avg ANLS": anls,
		"Avg retrieval precision": answ_page_pred_acc,
		"Scores by samples": eval_res["scores_by_samples"],
		"Dataset size": f"{data_size*100}%",
		"Inference time": inf_time,
		"Avg load time": avg_load_time,
		"Avg retrieval time": avg_retrieval_time,
		"Avg generation time": avg_generation_time,
		"Total load time": f"{total_load_time} ({eval_res['total_load_time']/inf_time_f*100:.2f}%)",
		"Total retrieval time": f"{total_retrieval_time} ({eval_res['total_retrieval_time']/inf_time_f*100:.2f}%)",
		"Total generation time": f"{total_generation_time} ({eval_res['total_generation_time']/inf_time_f*100:.2f}%)",
	}
	experiment_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	results_file = os.path.join(config["save_dir"], "results", "{:}_{:}_{:}.json".format(
		config.get("embed_model", "").lower(), config.get("page_retrieval", "").lower(), experiment_date)
	)
	save_json(results_file, save_data)
	print("Results correctly saved in: {:s}".format(results_file))
