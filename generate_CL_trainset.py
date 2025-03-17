import os
import time
import argparse
import gc
import torch
import sqlite3
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.MP_DocVQA import mpdocvqa_collate_fn
from src.metrics import Evaluator
from src.utils import load_config
from src.build_utils import build_model, build_dataset
from src.RAGVT5 import RAGVT5


def build_CL_trainset(
		data_loader: DataLoader,
		model: RAGVT5,
		evaluator: Evaluator,
		db_file_path: str,
		**kwargs
):
	print("Building the training set for CL...")
	model.eval()
	conn = sqlite3.connect(db_file_path)
	cursor = conn.cursor()
	cursor.execute("DROP TABLE IF EXISTS trainset")
	cursor.execute("""
			CREATE TABLE trainset (
				id INTEGER PRIMARY KEY,
				anchor TEXT,
				positive TEXT
			)""")

	# Evaluate each batch
	for b, batch in enumerate(tqdm(data_loader)):
		try:
			bs = len(batch["question_id"])

			# Inference using the model
			_, pred_answers, _, pred_answers_conf, retrieval = model.inference(
				batch,
				return_retrieval=True
			)
			pred_answer_pages = retrieval["page_indices"]
			questions = batch["questions"] # (bs,)
			top_k_chunks = retrieval["text"] # (bs, k)

			# Compute metrics (accuracy, ANLS)
			metrics = []
			for b in range(bs):
				preds = pred_answers[b]
				if preds is None:
					metrics.append({"accuracy": [0]*len(top_k_chunks[b]), "anls": [0]*len(top_k_chunks[b])})
					continue
				gt_answers = [batch["answers"][b]]*len(preds)
				answer_types = batch.get("answer_type", None)[b] if batch.get("answer_type", None) is not None else None
				metrics.append(evaluator.get_metrics(gt_answers, preds, answer_types))
			
			# Prepare the pairs
			chunks_anls = [metrics[b]["anls"] for b in range(bs)] # (bs, k)
			good_chunks = []
			# bad_chunks = []
			for b in range(bs):
				good_chunks.append([chunk for chunk, anls in zip(top_k_chunks[b], chunks_anls[b]) if anls > 0.8])
				# bad_chunks.append([chunk for chunk, anls in zip(top_k_chunks[b], chunks_anls[b]) if anls < 0.2])
			
			for b in range(bs):
				anchor = questions[b]
				positives = good_chunks[b]
				for positive in positives:
					cursor.execute("INSERT INTO trainset (anchor, positive) VALUES (?, ?)", (anchor, positive))
		except Exception as e:
			print(f"Error processing batch {b}: {e}")
			continue		

		# Free memory
		del pred_answers, pred_answer_pages, pred_answers_conf, metrics, batch
		gc.collect()
		torch.cuda.empty_cache()

	# Close the connection
	conn.commit()
	conn.close()


if __name__ == "__main__":
	# Prepare model and dataset
	args = {
		"model": "RAGVT5",
		"dataset": "DUDE",
		"embed_model": "BGE", # BGE / VT5 / JINA
		"reranker_model": "BGE",
		"page_retrieval": "AnyConf",
		"add_sep_token": False,
		"batch_size": 130, # 50 Oracle / Concat / MajorPage / WeightMajorPage / AnyConfOracle, 32 MaxConf / AnyConf, 16 MaxConfPage / AnyConfPage
		"chunk_num": 20,
		"chunk_size": 60,
		"chunk_size_tol": 0.2,
		"overlap": 10,
		"include_surroundings": 0,
		# "model_weights": "/data3fast/users/elopez/checkpoints/ragvt5_concat_mp-docvqa_train_generator/best.ckpt",
		"embed_weights": "BAAI/bge-small-en-v1.5", # or VT5
		"reranker_weights": "BAAI/bge-reranker-v2-m3",
		"reorder_chunks": False,
		"rerank_filter_tresh": 0,
		"rerank_max_chunk_num": 10,
		"rerank_min_chunk_num": 1
	}
	extra_args = {
		"visible_devices": "3",
		"save_folder": "18-dude",
		"save_name_append": "",
		"val_size": 1.0,
		"log_wandb": True,
		"log_media_interval": 10,
		"return_scores_by_sample": True,
		"return_answers": True,
		"save_results": False,
		"save_continuously": True,
		"compute_stats": False,
		"compute_stats_examples": False,
		"n_stats_examples": 5,
	}
	args.update(extra_args)
	os.environ["CUDA_VISIBLE_DEVICES"] = args["visible_devices"]

	args = argparse.Namespace(**args)
	config = load_config(args)
	start_time = time.time()

	print("Building model...")
	config["page_retrieval"] = config["page_retrieval"].lower()
	model = build_model(config)
	model.to(config["device"])
	print("Building dataset...")
	data_size = 1.0
	dataset = build_dataset(config, split="train", size=data_size)
	train_data_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=mpdocvqa_collate_fn, num_workers=0)

	# Build the training set
	db_file_path = "/data/users/elopez/dude/cl_trainset.db"
	evaluator = Evaluator(config, case_sensitive=False)
	build_CL_trainset(
		train_data_loader,
		model,evaluator,
		db_file_path=db_file_path
	)
