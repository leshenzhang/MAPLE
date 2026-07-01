from __future__ import annotations
"""UniTS-Gen ML-diffusion TS-guess front-end — native MAPLE built-in (training-free).

Unlike GeodesicTSGuess (physics init from R+P geometries), UniTS-Gen needs only a
reaction SMILES + 0-based reactive atom indices; a pretrained HiEGNN SE(3)-equivariant
diffusion model (chemrxiv 10001667) samples N 3D TS guesses. Output Atoms feed straight
into Molecules -> BatchPRFO for refinement + freq — same downstream contract as geodesic.

Native: runs in MAPLE's cxtorch env (torch>=2.6) with NO torch_scatter / torch_cluster /
molop / qcbot / openbabel (vendored units/ carries native shims; heavy I/O deps are lazy).
Weights are an external ~267 MB asset (NOT in-repo): set UNITSGEN_MODEL_DIR or pass
model_dir (ModelScope XuLiCheng2025/UniTS-Gen-v1 -> best_full_model.pth).
"""
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ase import Atoms

from .logger import log_info
from ...jobABC import JobABC

_VENDOR = os.path.join(os.path.dirname(__file__), "unitsgen_vendor")
_DEFAULT_MODEL_DIR = os.environ.get(
    "UNITSGEN_MODEL_DIR",
    os.path.join(_VENDOR, "units", "model_path", "units_hiegnn"),
)
_MODEL_CACHE: dict = {}


def _install_torch_shims():
    """torch>=2.6 defaults weights_only=True; e3nn constants.pt + the CUDA-saved HiEGNN
    checkpoint are trusted local assets. Restore weights_only=False + map_location for
    CPU nodes. Idempotent."""
    import torch
    try:
        torch.serialization.add_safe_globals([slice])
    except Exception:
        pass
    if getattr(torch.load, "_unitsgen_patched", False):
        return
    _orig = torch.load

    def _patched(*a, **k):
        k.setdefault("weights_only", False)
        if not torch.cuda.is_available():
            k.setdefault("map_location", torch.device("cpu"))
        return _orig(*a, **k)

    _patched._unitsgen_patched = True
    torch.load = _patched


def _ensure_vendor_on_path():
    if _VENDOR not in sys.path:
        sys.path.insert(0, _VENDOR)


def _get_model(model_dir, device):
    key = (model_dir, str(device))
    if key not in _MODEL_CACHE:
        _ensure_vendor_on_path()
        from units.generate import load_model
        _MODEL_CACHE[key] = load_model(model_dir, ckpt_file="best_full_model.pth", device=device)
    return _MODEL_CACHE[key]


@dataclass
class UniTSGenParams:
    n_samples: int = 10
    charge: int = 0
    multi: int = 1
    model_dir: Optional[str] = None
    device: Optional[str] = None
    seed: Optional[int] = None
    resample: bool = False
    resample_steps: int = 10
    start_step: int = 40
    jump_len: int = 2


def generate_ts_guesses(smiles: str, reactive_atom_idx, charge: int = 0, multi: int = 1,
                        n_samples: int = 10, model_dir: Optional[str] = None,
                        device=None, seed: Optional[int] = None, resample: bool = False,
                        resample_steps: int = 10, start_step: int = 40,
                        jump_len: int = 2) -> List[Atoms]:
    """Reaction SMILES + 0-based reactive atom indices -> list[ase.Atoms] TS guesses
    (length n_samples). Pretrained-inference only (no training)."""
    _install_torch_shims()
    import torch
    _ensure_vendor_on_path()
    device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model_dir = model_dir or _DEFAULT_MODEL_DIR
    ckpt = os.path.join(model_dir, "best_full_model.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"UniTS-Gen weights not found: {ckpt}. External ~267 MB asset; set "
            "UNITSGEN_MODEL_DIR or pass model_dir (ModelScope XuLiCheng2025/UniTS-Gen-v1).")
    from units.data import gen_dataset_from_smiles
    try:
        from torch_geometric.loader import DataLoader
    except Exception:  # older PyG
        from torch_geometric.data import DataLoader

    args, model = _get_model(model_dir, device)
    if seed is not None:
        from units.utils import set_global_seed
        set_global_seed(seed)

    lst = [[smiles, list(reactive_atom_idx)] for _ in range(n_samples)]
    guesses: List[Atoms] = []
    with tempfile.TemporaryDirectory(prefix="unitsgen_") as tmp:
        ds = gen_dataset_from_smiles(None, smiles_react_atom_index_lst=lst, args=args,
                                     charge=charge, multi=multi, tag="unitsgen",
                                     root=tmp, ts_type="units")
        dl = DataLoader(ds, batch_size=n_samples, shuffle=False, num_workers=0)
        with torch.no_grad():
            for data in dl:
                data = data.to(device)
                x_traj, mol_atoms, node_mask = model.sample_traj(
                    data, fix_noise=False, resample=resample,
                    resample_steps=resample_steps, start_step=start_step, jump_len=jump_len)
                final = x_traj[-1]
                for b in range(final.shape[0]):
                    nm = node_mask[b].squeeze(-1).bool().cpu().numpy()
                    nums = mol_atoms[b].squeeze(-1).cpu().numpy()[nm]
                    coords = np.asarray(final[b])[nm]
                    guesses.append(Atoms(numbers=[int(z) for z in nums],
                                         positions=np.asarray(coords, dtype=float)))
    return guesses


class UniTSGenGuess(JobABC):
    """JobABC front-end mirroring GeodesicTSGuess.run() contract (produces ts_guess Atoms
    that feed Molecules -> BatchPRFO). Input = reaction SMILES + 0-based reactive indices."""

    def __init__(self, output: str, smiles: Optional[str] = None,
                 reactive_atom_idx=None, paras: Optional[dict] = None):
        super().__init__(output)
        self.smiles = smiles
        self.reactive_atom_idx = reactive_atom_idx
        self.params = self._init_params(paras)
        self.ts_guesses: List[Atoms] = []
        self.ts_guess: Optional[Atoms] = None

    @staticmethod
    def _init_params(paras: Optional[dict]) -> UniTSGenParams:
        p = UniTSGenParams()
        if paras:
            for f in UniTSGenParams().__dataclass_fields__:
                if f in paras:
                    setattr(p, f, paras[f])
        return p

    def run(self):
        if not self.smiles or self.reactive_atom_idx is None:
            raise ValueError("UniTSGenGuess needs `smiles` + `reactive_atom_idx` (0-based list).")
        p = self.params
        guesses = generate_ts_guesses(
            self.smiles, self.reactive_atom_idx, charge=p.charge, multi=p.multi,
            n_samples=p.n_samples, model_dir=p.model_dir, device=p.device, seed=p.seed,
            resample=p.resample, resample_steps=p.resample_steps,
            start_step=p.start_step, jump_len=p.jump_len)
        base, _ = os.path.splitext(self.output)
        from ase.io import write as _ase_write
        for i, g in enumerate(guesses):
            _ase_write(f"{base}_unitsgen_tsguess_{i}.xyz", g)
        natoms = len(guesses[0]) if guesses else 0
        log_info([f"[unitsgen] {len(guesses)} TS guesses ({natoms} atoms) -> "
                  f"{base}_unitsgen_tsguess_*.xyz\n"], self.output)
        self.ts_guesses = guesses
        self.ts_guess = guesses[0] if guesses else None
        return guesses
