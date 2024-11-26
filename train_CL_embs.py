import os
import torch
import sqlite3
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses


class SQLiteDataset(Dataset):
	def __init__(self, db_file):
		self.conn = sqlite3.connect(db_file)
		self.cursor = self.conn.cursor()
		# Get dataset size
		self.cursor.execute("SELECT COUNT(*) FROM pairs")
		self.size = self.cursor.fetchone()[0]

	def __len__(self):
		return self.size

	def __getitem__(self, idx):
		self.cursor.execute("SELECT anchor, positive FROM pairs WHERE id = ?", (idx + 1,))
		row = self.cursor.fetchone()
		if row is None:
			raise IndexError(f"Index {idx} out of range")
		return {"anchor": row[0], "positive": row[1]}

	def close(self):
		self.conn.close()


def train_CL_embs(
		model: SentenceTransformer,
		dataset: SQLiteDataset,
		**kwargs
):
	print("Training the embeddings for CL...")
	model.train()
	criterion = losses.MultipleNegativesRankingLoss(model)
	output_dir = kwargs.get("output_dir")
	
	args = SentenceTransformerTrainingArguments(
		output_dir=output_dir,
		per_device_train_batch_size=64,
		num_train_epochs=1,
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
	os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5"
	db_file_path = "/data3fast/users/elopez/data/cl_trainset.db"
	embed_model_name = "BAAI/bge-small-en-v1.5"
	cache_dir = "/data3fast/users/elopez/models"
	print("Loading embedding model...")
	embed_model = SentenceTransformer(embed_model_name, cache_folder=cache_dir)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	embed_model.to(device)
	print("Using device:", device)
	print("Loading trainset...")
	dataset = SQLiteDataset(db_file_path)

	# Train the embeddings
	train_CL_embs(embed_model, dataset, output_dir="/data3fast/users/elopez/models/bge-finetuned")
	dataset.close()
	print("Done!")
