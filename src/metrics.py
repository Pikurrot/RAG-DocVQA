import json
import os
import editdistance
import numpy as np
import re
from typing import Dict, List, Optional, Union
from .eval.extract_answer import extract_answer
from .eval.eval_score import eval_score, show_results, eval_acc_and_f1
from src._modules import get_layout_model_map
from src.utils import get_similarity_score

class Evaluator:
	def __init__(
		self,
		config: dict,
		case_sensitive: bool = False,
		model_name: str = "gpt-4o",
		prompt_path: Optional[str] = None
	):
		self.layout_map = get_layout_model_map(config)
		self.case_sensitive = case_sensitive
		self.get_edit_distance = editdistance.eval
		self.anls_threshold = 0.5
		self.model_name = model_name
		self.prompt = self._load_prompt(prompt_path) if prompt_path else None

		self.best_accuracy = 0
		self.best_epoch = 0

	def _load_prompt(self, prompt_path: str) -> str:
		"""Load the prompt template from file."""
		with open(prompt_path, 'r') as f:
			return f.read().strip()

	def get_metrics(
		self,
		gt_answers: List[List[str]],
		preds: Union[List[str], List[List[str]]],
		answer_types: List[str] = None,
		top_k_layout_labels: List[List[int]] = None
	) -> dict:
		if preds is None:
			return {"accuracy": [0]*len(gt_answers), "anls": [0]*len(gt_answers)}
		
		answer_types = answer_types if answer_types is not None else ["string" for _ in range(len(gt_answers))]
		batch_accuracy = []
		batch_anls = []
		layout_labels_accuracy = {value: [] for value in self.layout_map.values()}
		layout_labels_anls = {value: [] for value in self.layout_map.values()}

		# Compute metrics for each batch element
		for b in range(len(preds)):
			gt = [self._preprocess_str(gt_elm) for gt_elm in gt_answers[b]]
			if isinstance(preds[b], list):  # in case of Anyconf
				any_pred = [self._preprocess_str(pred) for pred in preds[b]]
				batch_accuracy_max = 0
				batch_anls_max = 0
				for i, pred in enumerate(any_pred):
					accuracy = self._calculate_accuracy(gt, pred, answer_types[b])
					anls = self._calculate_anls(gt, pred, answer_types[b])
					batch_accuracy_max = max(batch_accuracy_max, accuracy)
					batch_anls_max = max(batch_anls_max, anls)
					if top_k_layout_labels is not None:
						label = self.layout_map[top_k_layout_labels[b][i]]
						layout_labels_accuracy[label].append(accuracy)
						layout_labels_anls[label].append(anls)
				batch_accuracy.append(batch_accuracy_max)
				batch_anls.append(batch_anls_max)
			else:  # others
				pred = self._preprocess_str(preds[b])
				accuracy = self._calculate_accuracy(gt, pred, answer_types[b])
				anls = self._calculate_anls(gt, pred, answer_types[b])
				if top_k_layout_labels is not None:
					layout_labels = top_k_layout_labels[b]
					for label in layout_labels:
						label = self.layout_map[label]
						layout_labels_accuracy[label].append(accuracy)
						layout_labels_anls[label].append(anls)
				batch_accuracy.append(accuracy)
				batch_anls.append(anls)

		return {
			"accuracy": batch_accuracy,
			"anls": batch_anls,
			"layout_labels_accuracy": layout_labels_accuracy,
			"layout_labels_anls": layout_labels_anls
		}

	def get_retrieval_metric(
		self,
		gt_answer_page: List[int],
		pred_answer_pages: Union[List[int], List[List[int]]]
	) -> list:
		if isinstance(pred_answer_pages[0], int):
			retrieval_precision = [
				1 if gt == pred else 0
				for gt, pred in zip(gt_answer_page, pred_answer_pages)
			]
		else:
			retrieval_precision = [
				1 if gt in preds else 0
				for gt, preds in zip(gt_answer_page, pred_answer_pages)
			]
		return retrieval_precision

	def eval_retrieval(
		self,
		batch: dict,
		retrieval: dict
	) -> dict:
		if not retrieval:
			return {"chunk_score": [0]*len(batch["answers"])}
		# Check if the answer is in the chunks
		chunks = retrieval["text"]  # (bs, k)
		answers = batch["answers"]  # (bs, n)
		scores = []  # (bs,)
		for b in range(len(answers)):
			top_chunks = chunks[b]
			possible_answers = answers[b]
			best_score = 0
			for ans in possible_answers:
				ans_scores = [get_similarity_score(chunk, ans) for chunk in top_chunks]
				best_score = max(best_score, max(ans_scores+[0]))
			scores.append(np.log(best_score+1) / np.log(2))
		
		return {
			"chunk_score": scores
		}

	def update_global_metrics(
		self,
		accuracy: float,
		anls: float,
		current_epoch: int
	):
		if accuracy > self.best_accuracy:
			self.best_accuracy = accuracy
			self.best_epoch = current_epoch
			return True
		else:
			return False

	def _preprocess_str(self, string: str) -> str:
		if string is None:
			return ""
		if not self.case_sensitive:
			string = string.lower()
		return string.strip()

	def _is_special_case(self, string: str) -> bool:
		# Check for special cases that require exact matching
		if "https://" in string or "http://" in string:
			return True
		if string.endswith((".py", ".ipynb")):
			return True
		if string.startswith("page"):
			return True
		if re.fullmatch(r'\b\d+(-\d+|\s\d+)?\b', string):
			return True
		if "a.m." in string or "p.m." in string:
			return True
		if re.fullmatch(r'\b\d{4}[-\s]\d{2}[-\s]\d{2}\b', string):
			return True
		if re.fullmatch(r'\b\d{4}[-\s]\d{2}\b', string):
			return True
		if re.fullmatch(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', string):
			return True
		return False

	def _calculate_accuracy(
		self,
		gt: List[str],
		pred: str,
		answer_type: str
	) -> int:
		if answer_type == "not-answerable":
			return 1 if pred in ["", "none", "NA", None, []] else 0

		if pred == "none" and answer_type != "not-answerable":
			return 0

		# Handle special cases that require exact matching
		if any(self._is_special_case(g) for g in gt):
			return 1 if pred in gt else 0

		# Handle different answer types
		if answer_type == "int":
			try:
				gt_int = [int(g) for g in gt]
				pred_int = int(float(pred))
				return 1 if pred_int in gt_int else 0
			except:
				return 0
		elif answer_type == "float":
			try:
				gt_float = [float(g.strip().rstrip("%")) for g in gt]
				pred_float = float(pred.strip().rstrip("%"))
				# Check if prediction is close to any ground truth
				return 1 if any(abs(pred_float - g) < 0.01 for g in gt_float) else 0
			except:
				return 0
		elif answer_type == "list":
			try:
				gt_list = [sorted(g.strip("[]").split(",")) for g in gt]
				pred_list = sorted(pred.strip("[]").split(","))
				return 1 if pred_list in gt_list else 0
			except:
				return 0
		else:  # string
			return 1 if pred in gt else 0

	def _calculate_anls(
		self,
		gt: List[str],
		pred: str,
		answer_type: str
	) -> float:
		if len(pred) == 0:
			return 0

		if answer_type == "not-answerable":
			return 1 if pred in ["", "none", "NA", None, []] else 0

		if pred == "none" and answer_type != "not-answerable":
			return 0

		# Handle special cases that require exact matching
		if any(self._is_special_case(g) for g in gt):
			return 1.0 if pred in gt else 0.0

		# Handle different answer types
		if answer_type in ["int", "float", "list"]:
			return 1.0 if self._calculate_accuracy(gt, pred, answer_type) == 1 else 0.0

		# For strings, use ANLS
		answers_similarity = [1 - self.get_edit_distance(gt_elm, pred) / max(len(gt_elm), len(pred)) for gt_elm in gt]
		max_similarity = max(answers_similarity)
		return max_similarity if max_similarity >= self.anls_threshold else 0.0

	def evaluate(self, predictions: List[Dict], output_path: Optional[str] = None) -> Dict:
		"""
		Evaluate predictions using MMLongBenchDoc methodology.
		
		Args:
			predictions: List of prediction dictionaries containing:
				- question: The question asked
				- answer: Ground truth answer
				- pred: Model prediction
				- answer_format: Type of answer (Int, Float, Str, List, None)
				- evidence_pages: List of pages containing evidence
				- evidence_sources: List of evidence sources
				- doc_type: Type of document
			output_path: Optional path to save detailed results
		
		Returns:
			Dictionary containing evaluation metrics
		"""
		# Extract structured answers using GPT-4o
		for pred in predictions:
			if self.prompt:
				extracted_answer = extract_answer(
					question=pred["question"],
					output=pred["pred"],
					prompt=self.prompt,
					model_name=self.model_name
				)
				pred["pred"] = extracted_answer

		# Calculate scores for each prediction
		for pred in predictions:
			pred["score"] = eval_score(
				gt=pred["answer"],
				pred=pred["pred"],
				answer_type=pred.get("answer_format", "Str")
			)

		# Generate detailed evaluation report
		if output_path:
			show_results(predictions, output_path)

		# Calculate overall metrics
		acc, f1 = eval_acc_and_f1(predictions)
		
		return {
			"accuracy": acc,
			"f1_score": f1,
			"num_samples": len(predictions)
		}

	def save_predictions(self, predictions: List[Dict], output_path: str):
		"""Save predictions to a JSON file."""
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		with open(output_path, 'w') as f:
			json.dump(predictions, f, indent=2)
