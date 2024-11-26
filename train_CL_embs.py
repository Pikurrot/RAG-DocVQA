import os
import torch
import sqlite3
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)


def train_CL_embs(
		model: SentenceTransformer,
		dataset: Dataset,
		**kwargs
):
	print("Training the embeddings for CL...")
	model.train()
	criterion = losses.MultipleNegativesRankingLoss(model)
	output_dir = kwargs.get("output_dir")
	
	args = SentenceTransformerTrainingArguments(
		output_dir=output_dir,
		per_device_train_batch_size=64,
		num_train_epochs=10,
		logging_steps=1,
		report_to="wandb"
	)
	trainer = SentenceTransformerTrainer(
		model=model,
		args=args,
		train_dataset=dataset,
		loss=criterion
	)
	trainer.train()


if __name__ == "__main__":
	# Prepare model and dataset
	os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(1,10))
	db_file_path = "/data3fast/users/elopez/data/cl_trainset.db"
	embed_model_name = "BAAI/bge-small-en-v1.5"
	cache_dir = "/data3fast/users/elopez/models"
	print("Loading embedding model...")
	embed_model = SentenceTransformer(embed_model_name, cache_folder=cache_dir)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	embed_model.to(device)
	print("Using device:", device)
	print("Loading trainset...")
	conn = sqlite3.connect(db_file_path)
	dataset = Dataset.from_sql(
		sql="SELECT anchor, positive FROM trainset",
		con=conn,
	)

	# Train the embeddings
	train_CL_embs(embed_model, dataset, output_dir="/data3fast/users/elopez/models/bge-finetuned")
	conn.close()
	print("Done!")
