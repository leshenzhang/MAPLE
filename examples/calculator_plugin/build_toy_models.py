"""Write two toy checkpoints next to this script so the example inputs can run.

    python build_toy_models.py

Produces ``toy_jit.pt`` (TorchScript) and ``toy_state.pt`` (plain state_dict),
consumed by ``sp_jit.inp`` and ``sp_pt.inp`` respectively. Replace with your own
real model files in production.
"""

import os

import torch

from my_calculator_plugin import ToyHarmonicModel

HERE = os.path.dirname(os.path.realpath(__file__))


def main():
    model = ToyHarmonicModel().double().eval()

    jit_path = os.path.join(HERE, 'toy_jit.pt')
    torch.jit.save(torch.jit.script(model), jit_path)

    pt_path = os.path.join(HERE, 'toy_state.pt')
    torch.save(
        {'hparams': {'k': 1.0, 'r0': 1.2, 'rcut': 3.0}, 'state_dict': model.state_dict()},
        pt_path,
    )

    print('wrote', jit_path)
    print('wrote', pt_path)


if __name__ == '__main__':
    main()
