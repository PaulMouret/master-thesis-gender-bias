# Implementation from with changes : https://github.com/tomlimi/DAMA

import torch
import numpy as np

import sys

from utils import nethook


def apply_dama_on_module(old_mlp, P, mu_in, mu_out, projection_location):

    # Apply DAMA on the module

    if projection_location == "before":
        old_mlp.weight = torch.nn.Parameter(old_mlp.weight @ P)
    elif projection_location == "after":
        old_mlp.weight = torch.nn.Parameter(P @ old_mlp.weight)
    else:
        raise ValueError("projection_location must be either 'before' or 'after'")

    return old_mlp


def load_dama_model(model, hparams, projection_file):
    layers = dict(model.named_modules())
    devices = [layers["model.layers.{}.mlp".format(i)].down_proj.weight.device for i in hparams.layers]
    #print(f"Loading projections from {projection_file}")
    loaded_projections = np.load(projection_file, allow_pickle=True).item()
    if torch.cuda.is_available():
        projections = {m_name: (torch.tensor(values['M'], device=dev, dtype=torch.float16),
                                torch.tensor(values['mu_in'], device=dev, dtype=torch.float16),
                                torch.tensor(values['mu_out'], device=dev, dtype=torch.float16))
                       for dev, (m_name, values) in zip(devices, loaded_projections.items())}
    else:
        projections = {m_name: (torch.tensor(values['M'], device='cpu', dtype=torch.float32),
                                torch.tensor(values['mu_in'], device='cpu', dtype=torch.float32),
                                torch.tensor(values['mu_out'], device='cpu', dtype=torch.float32))
                       for m_name, values in loaded_projections.items()}

    with torch.no_grad():
        for m_name, (P, mu_in, mu_out) in projections.items():
            if int(m_name.split('.')[2]) not in hparams.layers:
                continue

            orig_module = nethook.get_module(model, m_name)
            new_module = apply_dama_on_module(orig_module, P, mu_in, mu_out,
                                              hparams.projection_location if hasattr(hparams, "projection_location") else "after")

            nethook.replace_module(model, m_name, new_module)

        #print(f"New weights successfully inserted into layers: {hparams.layers}")

    return model
