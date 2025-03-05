import os
import torch
import sqlite3
from datasets import Dataset
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)


class CLIPStyleLoss(nn.Module):
	def __init__(self, model, temperature=0.07):
		super(CLIPStyleLoss, self).__init__()
		self.model = model
		self.temperature = temperature
		self.cross_entropy_loss = nn.CrossEntropyLoss()

	def forward(self, sentence_features, labels=None):
		embeddings_a = self.model(sentence_features[0])['sentence_embedding']
		embeddings_b = self.model(sentence_features[1])['sentence_embedding']

		# Normalize embeddings
		embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
		embeddings_b = F.normalize(embeddings_b, p=2, dim=1)

		# Compute cosine similarity
		logits = torch.matmul(embeddings_a, embeddings_b.t()) / self.temperature

		# Create labels
		labels = torch.arange(len(logits)).to(logits.device)

		# Compute loss
		loss_a = self.cross_entropy_loss(logits, labels)
		loss_b = self.cross_entropy_loss(logits.t(), labels)
		loss = (loss_a + loss_b) / 2
		return loss


def train_CL_embs(
		model: SentenceTransformer,
		dataset: Dataset,
		**kwargs
):
	print("Training the embeddings for CL...")
	model.train()
	criterion = losses.MultipleNegativesRankingLoss(model)
	# criterion = CLIPStyleLoss(model)
	output_dir = kwargs.get("output_dir")
	
	args = SentenceTransformerTrainingArguments(
		output_dir=output_dir,
		per_device_train_batch_size=1,
		num_train_epochs=10,
		logging_steps=1,
		report_to="wandb",
		fp16=True,
		gradient_accumulation_steps=4,
		gradient_checkpointing=True
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
	os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(4,10))
	db_file_path = "/data3fast/users/elopez/data/cl_trainset.db"
	embed_model_name = "BAAI/bge-m3"
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
	train_CL_embs(embed_model, dataset, output_dir="/data3fast/users/elopez/models/bge-m3-finetuned")
	conn.close()
	print("Done!")
