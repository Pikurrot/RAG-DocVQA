import socket
import datetime
import wandb as wb
import numpy as np
import matplotlib.pyplot as plt
from src.RAGVT5 import RAGVT5

class Logger:

	def __init__(self, config: dict, experiment_name: str):
		self.log_wandb = config["log_wandb"]
		self.log_folder = config["save_dir"]
		self.experiment_name = experiment_name

		machine_dict = {"cvc117": "Local", "cudahpc16": "DAG", "cudahpc25": "DAG-A40"}
		machine = machine_dict.get(socket.gethostname(), socket.gethostname())

		dataset = config["dataset_name"]
		page_retrieval = config.get("page_retrieval", "-").capitalize()
		visual_encoder = config.get("visual_module", {}).get("model", "-").upper()

		document_pages = config.get("max_pages", None)
		page_tokens = config.get("page_tokens", None)
		tags = [config["model_name"], dataset, machine]
		config = {"Model": config["model_name"], "Weights": config["model_weights"], "Dataset": dataset,
				"Page retrieval": page_retrieval, "Visual Encoder": visual_encoder,
				"Batch size": config["batch_size"], "Max. Seq. Length": config.get("max_sequence_length", "-"),
				"lr": config["lr"], "seed": config["seed"]}

		if document_pages:
			config["Max Pages"] = document_pages

		if page_tokens:
			config["PAGE tokens"] = page_tokens

		self.logger = wb.init(project="RAG-DocVQA", name=self.experiment_name, dir=self.log_folder, tags=tags, config=config)
		self._print_config(config)

		self.current_epoch = 0
		self.len_dataset = 0

	def _print_config(self, config: dict):
		print("{:s}: {:s} \n{{".format(config["Model"], config["Weights"]))
		for k, v in config.items():
			if k != "Model" and k != "Weights":
				print("\t{:}: {:}".format(k, v))
		print("}\n")

	def log(self, *args, **kwargs):
		if self.log_wandb:
			self.logger.log(*args, **kwargs)

	def log_model_parameters(self, model: RAGVT5):
		total_params = sum(p.numel() for p in model.generator.parameters())
		trainable_params = sum(p.numel() for p in model.generator.parameters() if p.requires_grad)

		self.logger.config.update({
			"Model Params": int(total_params / 1e6),  # In millions
			"Model Trainable Params": int(trainable_params / 1e6)  # In millions
		})

		print("Model parameters: {:d} - Trainable: {:d} ({:2.2f}%)".format(
			total_params, trainable_params, trainable_params / total_params * 100))

	def log_val_metrics(
			self,
			accuracy: float,
			anls: float,
			retrieval_precision: float,
			avg_chunk_score: float,
			update_best: bool=False
	):

		str_msg = "Epoch {:d}: Accuracy {:2.4f}     ANLS {:2.4f}    Retrieval precision: {:2.4f}   Avg. chunk score: {:2.4f}"\
			.format(self.current_epoch, accuracy, anls, retrieval_precision, avg_chunk_score)
		self.logger.log({
			"Val/Epoch Accuracy": accuracy,
			"Val/Epoch ANLS": anls,
			"Val/Epoch Ret. Prec": retrieval_precision,
			"Val/Epoch Chunk Score": avg_chunk_score
		}, step=self.current_epoch*self.len_dataset + self.len_dataset)

		if update_best:
			str_msg += "\tBest Accuracy!"
			self.logger.config.update({
				"Best Accuracy": accuracy,
				"Best epoch": self.current_epoch
			}, allow_val_change=True)

		print(str_msg)

class LoggerEval:
	
	def __init__(self, config: dict, experiment_name: str, log_media_interval: int = 1):
		self.log_wandb = config["log_wandb"]
		self.log_folder = config["save_dir"]
		self.experiment_name = experiment_name
		self.log_media_interval = log_media_interval
		self.log_media_counter = 0
		if self.log_wandb:
			self.logger = wb.init(project="RAG-DocVQA-Eval", name=self.experiment_name, dir=self.log_folder, config=config)
		self._print_config(config)
	
	def _print_config(self, config: dict):
		for k, v in config.items():
			if k != "Model" and k != "Weights":
				print("\t{:}: {:}".format(k, v))
		print("}\n")

	def log(self, *args, **kwargs):
		if self.log_wandb:
			self.logger.log(*args, **kwargs)

	def parse_and_log(self, log_data):
		self.log_media_counter += 1
		for key, value in log_data.items():
			if isinstance(value, (int, float)):
				# Log numerical value directly
				self.log({key: value})
			elif isinstance(value, dict) and "values" in value and "config" in value:
				if (self.log_media_counter == self.log_media_interval) or (self.log_media_counter == -1):
					self.log_media_counter = -1
					chart_type = value["config"].get("chart_type", "default")
					if chart_type == "pie":
						self.log_pie_chart(key, value["values"])
					elif chart_type == "spider":
						self.log_spider_chart(key, value["values"], value["config"].get("legend"), value["config"].get("log_scale", False))
					else:
						print(f"Unsupported chart type: {chart_type} for key {key}")
		if (self.log_media_counter == self.log_media_interval) or (self.log_media_counter == -1):
			self.log_media_counter = 0

	def log_pie_chart(self, key, values):
		labels = list(values.keys())
		sizes = list(values.values())
		if all(size == 0 for size in sizes):
			sizes = [1 for _ in sizes]
		
		fig, ax = plt.subplots()
		ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 14})
		ax.axis('equal')
		
		self.log({key: wb.Image(fig)})
		plt.close(fig)

	def log_spider_chart(self, key, values_list, legend=None, log_scale=False):
		num_vars = len(values_list[0])  # Number of categories
		angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
		angles += angles[:1]  # Close the loop
		
		fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
		
		if legend is None:
			legend = [str(i + 1) for i in range(len(values_list))]
		
		for values, label in zip(values_list, legend):
			categories = list(values.keys())
			data = list(values.values())
			data += data[:1]  # Close the loop
			
			ax.plot(angles, data, linewidth=2, label=label)
			ax.fill(angles, data, alpha=0.3)
		
		if log_scale:
			ax.set_yscale("log")
		ax.set_xticks(angles[:-1])
		ax.set_xticklabels(categories, fontsize=14)
		ax.legend(loc="upper left", fontsize=14, bbox_to_anchor=(0.5, -0.1))
		
		self.log({key: wb.Image(fig)})
		fig.subplots_adjust(bottom=0.2)
		plt.close(fig)
