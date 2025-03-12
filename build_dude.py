import argparse
from src.DUDE import build_dude
from src.utils import load_config

args = {
	"model": "RAGVT5",
	"dataset": "DUDE",
	"page_retrieval": "Concat",
}

args = argparse.Namespace(**args)
config = load_config(args)
dataset = build_dude(config, split="val")
