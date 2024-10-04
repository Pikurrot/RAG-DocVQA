import os
import time
import datetime
import argparse
import numpy as np
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

	all_pred_answers = []
	model.eval()

	# Evaluate each batch
	for b, batch in enumerate(tqdm(data_loader)):
		bs = len(batch["question_id"])

		# Inference using the model
		outputs, pred_answers, pred_answer_pages, pred_answers_conf, retrieval = \
			model.inference(batch, return_retrieval=False, include_surroundings=10, k=5)

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

		if return_answers:
			all_pred_answers.extend(pred_answers)

	if not return_scores_by_sample:
		# Compute average metrics
		total_accuracies = total_accuracies/len(data_loader.dataset)
		total_anls = total_anls/len(data_loader.dataset)
		total_ret_prec = total_ret_prec/len(data_loader.dataset)
		scores_by_samples = []

	return total_accuracies, total_anls, total_ret_prec, all_pred_answers, scores_by_samples


if __name__ == "__main__":
	# Prepare model and dataset
	args = {
		"model": "VT5",
		"dataset": "MP-DocVQA",
		"embed_model": "BGE" # VT5 or BGE
	}
	args = argparse.Namespace(**args)
	config = load_config(args)
	start_time = time.time()
	print("Building model...")
	model = build_model(config)
	model.to(config["device"])
	print("Building dataset...")
	dataset = build_dataset(config, split="val")
	val_data_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=mpdocvqa_collate_fn)

	# Evaluate the model
	print("Evaluating...")
	evaluator = Evaluator(case_sensitive=False)
	accuracy_list, anls_list, answer_page_pred_acc_list, pred_answers, scores_by_samples = \
		evaluate(val_data_loader, model, evaluator, return_scores_by_sample=True, return_answers=True)
	accuracy = np.mean(accuracy_list)
	anls = np.mean(anls_list)
	answ_page_pred_acc = np.mean(answer_page_pred_acc_list)
	
	# Save results
	inf_time = time_stamp_to_hhmmss(time.time() - start_time, string=True)
	save_data = {
		"Model": config["model_name"],
		"Model_weights": config["model_weights"],
		"Dataset": config["dataset_name"],
		"Page retrieval": config.get("page_retrieval", "-").capitalize(),
		"Inference time": inf_time,
		"Mean accuracy": accuracy,
		"Mean ANLS": anls,
		"Mean Retrieval precision": answ_page_pred_acc,
		"Scores by samples": scores_by_samples,
	}
	experiment_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	results_file = os.path.join(config["save_dir"], "results", "{:}_{:}_{:}.json".format(
		config.get("embed_model", "").lower(), config.get("page_retrieval", "").lower(), experiment_date)
	)
	save_json(results_file, save_data)
	print("Results correctly saved in: {:s}".format(results_file))
