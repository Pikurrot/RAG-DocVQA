import socket
import datetime
import wandb as wb
from src.RAG_VT5 import RAGVT5

class Logger:

    def __init__(self, config: dict):

        self.log_folder = config["save_dir"]

        experiment_date = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        self.experiment_name = "{:s}__{:}".format(config["model_name"], experiment_date)

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

