import os
import random
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
from typing import Any, List, Tuple, Dict
from torch.utils.data import Dataset
from time import time
from src.process_pdf import load_pdf

class MMLongBenchDoc(Dataset):

    def __init__(
            self,
            config: dict,
    ):
        data_dir = config["data_dir"]
        samples_path = os.path.join(data_dir, "data", "samples.json")
        documents_dir = os.path.join(data_dir, "data", "documents")
        page_retrieval = config.get("page_retrieval", "concat")
        split = config.get("split", "all")  # Assuming the dataset might have splits
        
        # Load samples from JSON
        with open(samples_path, 'r') as f:
            self.samples = json.load(f)
            
        # Apply split if needed
        if split != "all":
            # Implement split logic if necessary
            pass
            
        size = config.get("size", 1.0)
        if isinstance(size, float) and size < 1.0:
            self.samples = self.samples[:int(size*len(self.samples))]
        elif isinstance(size, tuple):
            self.samples = self.samples[int(size[0]*len(self.samples)):int(size[1]*len(self.samples))]

        self.page_retrieval = page_retrieval.lower()
        assert(self.page_retrieval in ["oracle", "concat", "logits", "custom", "maxconf", "anyconf", "maxconfpage", "anyconfpage", "majorpage", "weightmajorpage", "anyconforacle"])

        self.documents_dir = documents_dir
        self.max_answers = 2
        
        # Cache for loaded documents
        self.document_cache = {}
        
        self.use_images = config.get("use_images", False)
        self.get_raw_ocr_data = config.get("get_raw_ocr_data", False)
        self.max_pages = config.get("max_pages", 1)
        
        # Preload documents if specified
        self.preload_documents = config.get("preload_documents", False)
        if self.preload_documents:
            doc_ids = list(set([sample["doc_id"] for sample in self.samples]))
            for doc_id in tqdm(doc_ids, desc="Preloading documents"):
                doc_path = os.path.join(self.documents_dir, doc_id)
                self.document_cache[doc_id] = load_pdf(doc_path)

    def __len__(self):
        return len(self.samples)

    def _load_document(self, doc_id: str) -> Dict:
        """Load document from file or cache"""
        if doc_id in self.document_cache:
            return self.document_cache[doc_id]
        
        doc_path = os.path.join(self.documents_dir, doc_id)
        document = load_pdf(doc_path)
        self.document_cache[doc_id] = document
        return document

    def sample(
            self,
            idx: int = None,
            question_id: str = None
    ) -> Dict[str, Any]:

        if idx is not None:
            return self.__getitem__(idx)

        if question_id is not None:
            for idx in range(self.__len__()):
                record = self.samples[idx]
                if record["id"] == question_id:  # Assuming the field is "id"
                    return self.__getitem__(idx)

            raise ValueError(f"Question ID {question_id} not in dataset.")

        idx = random.randint(0, self.__len__() - 1)
        return self.__getitem__(idx)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        start_time = time()
        record = self.samples[idx]

        question = record["question"]
        answers = [record.get("answer", "").lower()]  # Assuming single answer
        doc_id = record["doc_id"]
        
        # Load document
        document = self._load_document(doc_id)
        ocr_tokens = document["ocr_tokens"]
        ocr_boxes = document["ocr_boxes"]
        
        num_pages = len(ocr_tokens)
        answer_page_idx = 0  # Default to first page if not specified
        
        # If the answer page is known, use it
        if "answer_page_idx" in record:
            answer_page_idx = record["answer_page_idx"]
        
        if self.page_retrieval in ["oracle", "anyconforacle"]:
            context = [" ".join([word.lower() for word in ocr_tokens[answer_page_idx]])]
            context_page_corresp = None
            num_pages_used = 1

            if self.use_images:
                images = [document["images"][answer_page_idx]]

            if self.get_raw_ocr_data:
                words = [[word.lower() for word in ocr_tokens[answer_page_idx]]]
                boxes = [ocr_boxes[answer_page_idx]]
            
            start_idxs, end_idxs = self._get_start_end_idx(context[0], answers)
        
        elif self.page_retrieval in ["concat", "logits", "maxconf", "anyconf", "maxconfpage", "anyconfpage", "majorpage", "weightmajorpage"]:
            context = []
            for page_ix in range(num_pages):
                context.append(" ".join([word.lower() for word in ocr_tokens[page_ix]]))

            context_page_corresp = None
            num_pages_used = num_pages

            if self.use_images:
                images = document["images"]

            if self.get_raw_ocr_data:
                words = []
                boxes = ocr_boxes
                for p in range(num_pages):
                    words.append([word.lower() for word in ocr_tokens[p]])
            
            start_idxs, end_idxs = self._get_start_end_idx(context[min(answer_page_idx, len(context)-1)], answers)

        elif self.page_retrieval == "custom":
            first_page, last_page = self.get_pages(num_pages, answer_page_idx)
            relative_answer_page_idx = answer_page_idx - first_page
            num_pages_used = min(last_page - first_page, self.max_pages)

            words = []
            boxes = []
            context = []
            
            for page_ix in range(first_page, last_page):
                if page_ix < num_pages:
                    words.append([word.lower() for word in ocr_tokens[page_ix]])
                    boxes.append(ocr_boxes[page_ix])
                    context.append(' '.join([word.lower() for word in ocr_tokens[page_ix]]))

            context_page_corresp = None

            # Pad if necessary
            if num_pages_used < self.max_pages:
                for _ in range(self.max_pages - num_pages_used):
                    words.append([''])
                    boxes.append(np.zeros([1, 4], dtype=np.float32))
                    context.append('')

            if self.use_images:
                images = [document["images"][page_ix] for page_ix in range(first_page, last_page) if page_ix < num_pages]
                # Pad with blank images if needed
                images += [Image.new('RGB', (2, 2)) for _ in range(self.max_pages - len(images))]
                
            start_idxs, end_idxs = None, None

        sample_info = {
            "question_id": record.get("id", idx),
            "questions": question,
            "contexts": context,
            "context_page_corresp": context_page_corresp,
            "answers": answers,
            "answer_page_idx": answer_page_idx,
            "num_pages": num_pages_used,
            "load_time": time()-start_time
        }

        if self.use_images:
            sample_info["images"] = images

        if self.get_raw_ocr_data:
            sample_info["words"] = words
            sample_info["boxes"] = boxes
        else:  # Information for extractive models
            sample_info["start_indxs"] = start_idxs
            sample_info["end_indxs"] = end_idxs

        return sample_info

    def _get_start_end_idx(
            self,
            context: str,
            answers: List[str]
    ) -> Tuple[int, int]:

        answer_positions = []
        for answer in answers:
            if not answer:
                continue
                
            start_idx = context.find(answer)

            if start_idx != -1:
                end_idx = start_idx + len(answer)
                answer_positions.append([start_idx, end_idx])

        if len(answer_positions) > 0:
            start_idx, end_idx = random.choice(answer_positions)
        else:
            start_idx, end_idx = 0, 0

        return start_idx, end_idx

    def get_pages(self, num_pages: int, answer_page_idx: int) -> Tuple[int, int]:
        if num_pages <= self.max_pages:
            first_page, last_page = 0, num_pages
        else:
            first_page_lower_bound = max(0, answer_page_idx-self.max_pages+1)
            first_page_upper_bound = min(answer_page_idx, num_pages-self.max_pages)
            first_page = random.randint(first_page_lower_bound, first_page_upper_bound)
            last_page = min(first_page + self.max_pages, num_pages)

            if last_page - first_page < self.max_pages:
                first_page = max(0, last_page - self.max_pages)

            assert (answer_page_idx in range(first_page, last_page))  # answer page is in selected range.
            assert (first_page >= 0)
            assert (last_page <= num_pages)

        return first_page, last_page

def mmlongbenchdoc_collate_fn(batch: List[dict]) -> dict:
    batch = {k: [dic[k] for dic in batch] for k in batch[0]}  # List of dictionaries to dict of lists.
    return batch
