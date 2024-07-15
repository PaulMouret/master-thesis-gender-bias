# Implementation from with changes : https://github.com/tomlimi/DAMA

import json
import typing
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(1, '/lnet/aic/personal/mouretp/master_thesis2')
#This insertion is ugly and should not be necessary, but for some reason it is on cluster

from causal_tracing_globals import *

#REMOTE_URL = f"{REMOTE_ROOT_URL}/data/dsets/known_1000.json"


class KnownsDataset(Dataset):
    def __init__(self, data_dir: str, name_dataset: str, *args, **kwargs):
        data_dir = Path(data_dir)
        known_loc = data_dir / f"{name_dataset}.json"
        # if not known_loc.exists():
        #     print(f"{known_loc} does not exist. Downloading from {REMOTE_URL}")
        #     data_dir.mkdir(exist_ok=True, parents=True)
        #     torch.hub.download_url_to_file(REMOTE_URL, known_loc)

        with open(known_loc, "r") as f:
            self.data = json.load(f)

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def shuffle(self, seed=92):
        random.Random(seed).shuffle(self.data)