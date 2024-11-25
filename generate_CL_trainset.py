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
from src.RAG_VT5 import RAGVT5


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
		bs = len(batch["question_id"])

		# Inference using the model
		_, pred_answers, _, pred_answers_conf, retrieval = model.inference(
			batch,
			return_retrieval=True,
			chunk_num=kwargs["chunk_num"],
			chunk_size=kwargs["chunk_size"],
			overlap=kwargs["overlap"],
			include_surroundings=kwargs["include_surroundings"]
		)
		pred_answer_pages = retrieval["page_indices"]

		# Compute metrics (accuracy, ANLS)
		metrics = []
		for b in range(bs):
			preds = pred_answers[b]
			gt_answers = [batch["answers"][b]]*len(preds)
			answer_types = batch.get("answer_type", None)[b] if batch.get("answer_type", None) is not None else None
			metrics.append(evaluator.get_metrics(gt_answers, preds, answer_types))
		
		# Prepare the pairs
		questions = batch["questions"] # (bs,)
		top_k_chunks = retrieval["text"] # (bs, k)
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
		"dataset": "MP-DocVQA",
		"embed_model": "BGE", # BGE, VT5, BGE-M3, BGE-reranker
		"page_retrieval": "AnyConfOracle",
		"add_sep_token": False,
		"batch_size": 32,
		"chunk_num": 10,
		"chunk_size": 60,
		"overlap": 10,
		"include_surroundings": 0,
		"visible_devices": "0",
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
	dataset = build_dataset(config, split="train", size=data_size)
	train_data_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=mpdocvqa_collate_fn, num_workers=0)

	# Build the training set
	db_file_path = "/data3fast/users/elopez/data/cl_trainset.db"
	evaluator = Evaluator(case_sensitive=False)
	build_CL_trainset(
		data_loader=train_data_loader,
		model=model,
		evaluator=evaluator,
		db_file_path=db_file_path,
		chunk_num=config.get("chunk_num", 10),
		chunk_size=config.get("chunk_size", 60),
		overlap=config.get("overlap", 10),
		include_surroundings=config.get("include_surroundings", 0)
	)
