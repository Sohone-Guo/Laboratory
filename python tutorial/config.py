# ----------------- config.py -----------------
from yacs.config import CfgNode as CN

_C = CN()
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda:9"

_C.DATASET = CN()
_C.DATASET.TRAIN = "/AI/Kesci/data/kesci_type_1_v2/train/"

# ----------------- .yaml -----------------
MODEL:
  DEVICE: 'cuda:8'

DATASET:
  TRAIN: "/AI/Kesci/data/kesci_type_1_v3/train/"
  
# ----------------- main() -----------------

import sys
sys.path.append('.')
import argparse
from config import cfg


# make the configure 
parser = argparse.ArgumentParser(description="ReID Baseline Training")
parser.add_argument(
    "--config_file", default="", help="path to config file", type=str
)
parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                    nargs=argparse.REMAINDER)

args = parser.parse_args()

if args.config_file != "":
    cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

