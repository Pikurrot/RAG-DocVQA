import argparse
from src.build_utils import build_dataset
from src.utils import load_config

args = {
	"model": "RAGVT5",
	"dataset": "DUDE",
	"page_retrieval": "Concat",
}

args = argparse.Namespace(**args)
config = load_config(args)
dataset = build_dataset(config, split="val")
