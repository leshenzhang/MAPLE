import os
from typing import List, Union

from ase import Atoms
from ..utility import Molecules

from maple.function.timer import timer


# ====================================================================== #
# OPT-IN GPU-batched calc acquisition (Task: ai-maple-gpu dispatch wiring)
#
# The single-structure job paths keep using the per-Atoms ASE calculator the
# engine attaches (the oracle). When a job is submitted with a list/Molecules,
# the batched optimizers / saddle searchers / LQA-IRC need a *batched*
# calculator implementing the (prepare / get_ef_gpu [+ get_efh_gpu] /
# set_coords_ / step_cart_ / backup_coords) contract. These module-level
# helpers resolve that batched calc, mirroring scan._resolve_batched_calc and
# sp._get_batch_calc. They live here so dispatcher.py, optimization.py, ts/ts.py
# and irc/irc.py share ONE acquisition path. Torch / fairchem imports stay lazy
# (inside the functions) so importing this module pulls no torch.
# ====================================================================== #
def batch_device_str(params) -> str:
    """Map a job's device setting to the 'cuda'|'cpu' token the batched calc /
    optimizers expect. Honors an explicit ``params['batch_device']``; else
    derives from ``params['device']`` ('gpu*'/'cuda*' -> cuda when available)."""
    import torch
    dev = str(params.get("batch_device") or params.get("device") or "").lower()
    if dev.startswith("gpu") or dev.startswith("cuda") or dev in ("", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def _is_batch_calc(calc) -> bool:
    """Duck-typed test: a batched calculator exposes prepare() + get_ef_gpu().

    Delegates to the shared predicate so the batch-calc contract lives in ONE
    place (dispatcher/_batch_calc_utils.py, mirrors BatchCalcABC)."""
    from ._batch_calc_utils import is_batch_calc
    return is_batch_calc(calc)


def _uma_task(params, attached_calc=None) -> str:
    """Resolve the UMA task for the batched calc: explicit model_options['task'],
    else the single calc's task_name, else 'omol'."""
    task = (params.get("model_options") or {}).get("task")
    if task:
        return str(task).lower()
    task = getattr(attached_calc, "task_name", None)
    return str(task).lower() if task else "omol"


def _derive_uma_checkpoint_path(params, attached_calc=None):
    """Locate the UMA .pt the engine's single calc uses, so the batched calc
    loads the IDENTICAL model. Mirrors SetCalculator/UMACalculator resolution:
    explicit checkpoint_path/model_path, else local '<calculator>/model/<size>.pt'.
    Returns None for non-UMA models (caller raises a clear error)."""
    model = str(params.get("model") or "").lower()
    mo = params.get("model_options") or {}
    ckpt = mo.get("checkpoint_path") or mo.get("model_path")
    if ckpt and os.path.isfile(str(ckpt)):
        return str(ckpt)
    is_uma = "uma" in model or (
        attached_calc is not None
        and type(attached_calc).__name__ == "UMACalculator")
    if not is_uma:
        return None
    from pathlib import Path
    import maple.function.calculator as _calcpkg
    size = str(mo.get("size") or "uma-s-1p1").lower()
    cand = Path(_calcpkg.__file__).parent / "model" / f"{size}.pt"
    return str(cand) if cand.exists() else None


def resolve_batched_calc(params, atoms_list, attached_calc=None):
    """Return a GPU-batched calculator for an OPT-IN batched (list/Molecules) job.

    Resolution order (first hit wins):
      1. ``params['batched_calc']``        -- a pre-built batched calc instance.
      2. ``attached_calc`` already batch-capable (duck-typed prepare+get_ef_gpu),
         e.g. a UMABatchCalc someone attached by hand (the sp pattern).
      3/4. registry-driven build for ANY registered backend via
         ``make_batch_calc(model_name, model_path=...)`` -- MACE / ANI / AIMNet2 /
         MACE-POL / decoupled build from ``params['batch_model_path']`` or their
         own local model dir. UMA keeps its dedicated fallback: derive the SAME
         UMA checkpoint the engine's single calc uses, so a plain '#model=uma(...)'
         multi-structure job needs NO extra params (the normal job-interface path).

    CAPABILITY GATE: the resolved calc is then checked against the job's ``#solv``
    request. No batch backend applies an implicit-solvent correction, so a solvated
    multi-structure OPT/TS/IRC job must FAIL FAST here rather than silently running in
    the gas phase (see _batch_calc_utils.reject_batched_implicit_solvent). Same
    fail-fast pattern as the PBC gate in BatchCalcABC.prepare and the coupled-calc B=1
    routing.
    """
    from ._batch_calc_utils import reject_batched_implicit_solvent
    calc = _build_batched_calc(params, atoms_list, attached_calc=attached_calc)
    return reject_batched_implicit_solvent(params, calc, context="batched job")


def _build_batched_calc(params, atoms_list, attached_calc=None):
    """Resolution body for resolve_batched_calc (see its docstring); ungated."""
    calc = params.get("batched_calc")
    if calc is not None:
        return calc
    if _is_batch_calc(attached_calc):
        return attached_calc

    import torch

    # Bare registry token (drop any '(...)' options tail) + UMA detection. Only
    # UMA models carry the 'uma' substring among the registered backends.
    model_str = str(params.get("model") or "").lower()
    model_name = model_str.split("(")[0].strip()
    is_uma = ("uma" in model_str) or (
        attached_calc is not None
        and type(attached_calc).__name__ == "UMACalculator")

    # Path 3/4 (non-UMA registered backends): route through the generic factory.
    if model_name and not is_uma:
        from ..calculator.batch_calculator_base import make_batch_calc
        return make_batch_calc(
            model_name,
            model_path=params.get("batch_model_path"),
            device=batch_device_str(params),
            dtype=params.get("batch_dtype", torch.float64),
            **(params.get("batch_model_options") or {}),
        )

    # UMA fallback (default + explicit '#model=uma(...)'): preserve the EXACT
    # checkpoint auto-derive + UMABatchCalc construction the single calc uses.
    model_path = params.get("batch_model_path") or _derive_uma_checkpoint_path(
        params, attached_calc)
    if not model_path:
        raise ValueError(
            "Batched (list/Molecules) job needs a batched calculator. Provide "
            "params['batched_calc'] or params['batch_model_path'], attach a "
            "batch-capable calculator, or use '#model=uma(...)' so the batched "
            "UMA checkpoint is auto-resolved from the local model directory.")
    from ..calculator.uma._uma_batch_calculator import UMABatchCalc
    return UMABatchCalc(
        str(model_path),
        device=batch_device_str(params),
        dtype=params.get("batch_dtype", torch.float64),
        task=params.get("batch_task") or _uma_task(params, attached_calc),
    )


class Dispatcher():
    def __init__(self):
        pass

    def __call__(self, commandcontrol: dict, jobtype: int, atoms: Union[Atoms, Molecules, List[Atoms]], output:str, extra:dict=None) -> None:

        """
        Dispatches the job based on the job type.
        Args:
            commandcontrol: CommandControl object
            jobtype: The type of job to be performed.
            atoms: The ASE Atoms object, Molecules object, or a list of Atoms objects.
            output: The path to the output file.
            extra: Extra parameters to be passed to the job.
        """

        self.output = output
        self.commandcontrol = commandcontrol
        self.set_throshould(atoms)
        if jobtype == 'opt':
            from .optimization import Optimization

            # OPT-IN batched optimization: a list/Molecules routes to a batched
            # optimizer inside Optimization.run(); a single Atoms is UNCHANGED.
            opt = Optimization(output=output, atoms=atoms, params=commandcontrol.params)
            opt.run()
            
        elif jobtype == 'sp':
            from .sp import SinglePoint

            sp_params = commandcontrol.params if hasattr(commandcontrol, "params") else commandcontrol

            # Handle trajectory/multiple structures
            if isinstance(atoms, Molecules):
                atoms_input = atoms.multiatoms
                sp = SinglePoint(output=output, atoms=atoms_input, paras=sp_params)
                sp.run()
            elif isinstance(atoms, list):
                sp = SinglePoint(output=output, atoms=atoms, paras=sp_params)
                sp.run()
            else:
                # Single structure (backward compatibility)
                sp = SinglePoint(output=output, atoms=atoms, paras=sp_params)
                sp.run()

        elif jobtype == 'scan':
            from .scan import Scan

            if extra is not None:
                if 'scan' not in extra:
                    raise ValueError('Constraints not provided for scan job')
            else:
                raise ValueError('Constraints not provided for scan job')
            
            if isinstance(atoms, (list, Molecules)):
                raise NotImplementedError('For scan job, only one Atoms object is allowed.')

            scan = Scan(output=output, atoms=atoms, method=commandcontrol.params.get('method'), constraints=extra['scan'], params=commandcontrol.params)
            scan.run()
            
        elif jobtype == 'freq':
            from .frequency import Frequency
            if isinstance(atoms, (list, Molecules)):
                raise NotImplementedError('For frequency job, only one Atoms object is allowed.')
            freq = Frequency(output=output, atoms=atoms, paras=commandcontrol.params)
            freq.run()
            
        elif jobtype == 'ts':
            from .ts import TransitionState
            
            # TS job allows a Molecules/list for multi-structure methods.
            if isinstance(atoms, (list, Molecules)):
                method = commandcontrol.params.get('method')
                if method in ['neb', 'string', 'autoneb']:
                    # NEB/String/AutoNEB consume the band: pass the internal list.
                    atoms_input = atoms.multiatoms if isinstance(atoms, Molecules) else atoms
                    ts = TransitionState(output=output, atoms=atoms_input, method=method, params=commandcontrol.params)
                    ts.run()
                    return
                elif method in ('prfo', 'dimer'):
                    # OPT-IN batched saddle search over the B structures
                    # (BatchPRFO / BatchDimer). Single-Atoms path UNCHANGED below.
                    ts = TransitionState(output=output, atoms=atoms, method=method, params=commandcontrol.params)
                    ts.run()
                    return
                else:
                    raise ValueError(f'Unknown TS method: {method}')

            # Single Atoms object (oracle path, unchanged)
            ts = TransitionState(output=output, atoms=atoms, method=commandcontrol.params.get('method'), params=commandcontrol.params)
            ts.run()
        
        elif jobtype == 'irc':
            from .irc import IRC

            method = commandcontrol.params.get('method')
            if isinstance(atoms, (list, Molecules)):
                # OPT-IN batched IRC: 'lqa' (LQABatch), 'gs' (GSBatch, the default
                # IRC integrator), 'hpc' (HPCBatch), 'eulerpc' (EulerPCBatch).
                # Single-Atoms path UNCHANGED.
                if method not in ('lqa', 'gs', 'hpc', 'eulerpc'):
                    raise NotImplementedError(
                        f"Batched IRC (list/Molecules) supports 'lqa'/'gs'/'hpc'/'eulerpc' only; got {method!r}.")
                irc = IRC(output=output, atoms=atoms, method=method, params=commandcontrol.params)
                irc.run()
            else:
                irc = IRC(output=output, atoms=atoms, method=method, params=commandcontrol.params)
                irc.run()

        elif jobtype == 'md':
            from .md.ensemble.nve import NVE
            from .md.ensemble.nvt import NVT
            from .md.ensemble.npt import NPT

            if isinstance(atoms, (list, Molecules)):
                raise NotImplementedError('For MD job, only one Atoms object is allowed.')

            if 'charge' not in atoms.info or 'mult' not in atoms.info:
                raise ValueError(
                    "MD requires explicit charge and multiplicity. "
                    "Provide either 'XYZ <charge> <mult> <path>' or an inline 'charge mult' line before coordinates."
                )

            ensemble = commandcontrol.params.get('ensemble', 'nve').lower()
            if ensemble == 'nve':
                md = NVE(output=output, atoms=atoms, paras=commandcontrol.params)
            elif ensemble == 'nvt':
                md = NVT(output=output, atoms=atoms, paras=commandcontrol.params)
            elif ensemble == 'npt':
                md = NPT(output=output, atoms=atoms, paras=commandcontrol.params)
            else:
                raise ValueError(f"Unknown MD ensemble: '{ensemble}'")
            md.run()
            
            
        else:
            try:
                raise NotImplementedError('Job type not implemented')
            except NotImplementedError as e:
                self.log_error(str(e))
                
    def set_throshould(self, atoms) -> None:
        """
        Sets the convergence throshould for the atoms object.

        Args:
            atoms: The ASE Atoms object, Molecules object, or list of Atoms.
        """
        
        # Geometry optimization convergence thresholds (Eh/Å and Å), Gaussian-style
        if self.commandcontrol.params.get('level') == 'extratight':
            self.commandcontrol.params['f_max_th'] = 0.00030
            self.commandcontrol.params['f_rms_th'] = 0.00020
            self.commandcontrol.params['dp_max_th'] = 0.00030
            self.commandcontrol.params['dp_rms_th'] = 0.00020

        elif self.commandcontrol.params.get('level') == 'tight':
            self.commandcontrol.params['f_max_th'] = 0.00085
            self.commandcontrol.params['f_rms_th'] = 0.00055
            self.commandcontrol.params['dp_max_th'] = 0.00110
            self.commandcontrol.params['dp_rms_th'] = 0.00075

        elif self.commandcontrol.params.get('level') == 'medium':
            self.commandcontrol.params['f_max_th'] = 0.00285
            self.commandcontrol.params['f_rms_th'] = 0.00190
            self.commandcontrol.params['dp_max_th'] = 0.00315
            self.commandcontrol.params['dp_rms_th'] = 0.00210

        elif self.commandcontrol.params.get('level') == 'loose':
            self.commandcontrol.params['f_max_th'] = 0.00380
            self.commandcontrol.params['f_rms_th'] = 0.00250
            self.commandcontrol.params['dp_max_th'] = 0.00600
            self.commandcontrol.params['dp_rms_th'] = 0.00400

        elif self.commandcontrol.params.get('level') == 'extraloose':
            self.commandcontrol.params['f_max_th'] = 0.00755
            self.commandcontrol.params['f_rms_th'] = 0.00500
            self.commandcontrol.params['dp_max_th'] = 0.01200
            self.commandcontrol.params['dp_rms_th'] = 0.00800

        elif self.commandcontrol.params.get('level') == 'superloose':
            self.commandcontrol.params['f_max_th'] = 0.08500
            self.commandcontrol.params['f_rms_th'] = 0.05500
            self.commandcontrol.params['dp_max_th'] = 0.14500
            self.commandcontrol.params['dp_rms_th'] = 0.09500

        else:
            # fallback to medium
            self.commandcontrol.params['f_max_th'] = 0.00285
            self.commandcontrol.params['f_rms_th'] = 0.00190
            self.commandcontrol.params['dp_max_th'] = 0.00315
            self.commandcontrol.params['dp_rms_th'] = 0.00210

        # Apply thresholds to atoms
        if isinstance(atoms, Molecules):
            # Apply to all atoms in Molecules object
            for atom in atoms.multiatoms:
                atom.f_max_th = self.commandcontrol.params['f_max_th']
                atom.f_rms_th = self.commandcontrol.params['f_rms_th']
                atom.dp_max_th = self.commandcontrol.params['dp_max_th']   
                atom.dp_rms_th = self.commandcontrol.params['dp_rms_th']
        elif isinstance(atoms, list):
            # Apply to all atoms in list
            for atom in atoms:
                atom.f_max_th = self.commandcontrol.params['f_max_th']
                atom.f_rms_th = self.commandcontrol.params['f_rms_th']
                atom.dp_max_th = self.commandcontrol.params['dp_max_th']   
                atom.dp_rms_th = self.commandcontrol.params['dp_rms_th']
        else:
            # Single Atoms object
            atoms.f_max_th = self.commandcontrol.params['f_max_th']
            atoms.f_rms_th = self.commandcontrol.params['f_rms_th']
            atoms.dp_max_th = self.commandcontrol.params['dp_max_th']   
            atoms.dp_rms_th = self.commandcontrol.params['dp_rms_th']



    def log_error(self, error_message: str) -> None:
        """
        Logs error messages to the output file.

        Args:
            error_message: The error message to log.
        """
        with open(self.output, 'a') as file:
            file.write(f"ERROR: {error_message}\n")

    def log_info(self, info_message: list) -> None:
        """
        Logs info messages to the output file.

        Args:
            info_message: The info message to log.
        """
        with open(self.output, 'a') as file:
            for info in info_message:   
                file.write(f"{info}")
