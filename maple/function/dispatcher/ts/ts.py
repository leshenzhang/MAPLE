import numpy as np
from typing import Union, List

from torch import Tensor
from ase import Atoms

from ..jobABC import JobABC

from maple.function.utility import Molecules
from maple.function.timer import timer

class TransitionState(JobABC):

    def __init__(self, output: str, atoms: Union[Atoms, Molecules, List[Atoms]], params: dict, method:str=None):
        super().__init__(output)
        self.atoms = atoms
        self.params = params
        self.method = method

    def run(self):

        with timer("Transition State Optimization"):
            if self.method is None:
                raise ValueError('Method is not provided.')

            elif self.method == 'prfo':
                # OPT-IN batched PRFO for a list/Molecules of TS guesses;
                # a single Atoms keeps the unchanged single-structure PRFO.
                if isinstance(self.atoms, (list, Molecules)):
                    self._run_batched_prfo()
                else:
                    from .algorithm import PRFO
                    prfo = PRFO(
                        atoms=self.atoms,
                        output=self.output,
                        paras=self.params,
                    )
                    prfo.run()

            elif self.method == 'neb':
                # NEB now accepts Molecules object
                if isinstance(self.atoms, Molecules):
                    from .algorithm import NEB
                    neb = NEB(
                        output=self.output,
                        atoms_or_molecules=self.atoms,
                        paras=self.params
                    )
                    neb.run()
                elif isinstance(self.atoms, list):
                    # Legacy support for list input
                    if len(self.atoms) < 2:
                        raise ValueError('For NEB method, you should provide at least two structures (initial and final states).')
                    from .algorithm import NEB
                    # Convert list to Molecules for NEB
                    molecules = Molecules(self.atoms)
                    neb = NEB(
                        output=self.output,
                        atoms_or_molecules=molecules,
                        paras=self.params
                    )
                    neb.run()
                else:
                    raise ValueError('For NEB method, you should provide a Molecules object or a list of at least two structures.')
                    
            elif self.method == 'string':
                # TODO: Update String/GSM to accept Molecules object instead of list
                if not isinstance(self.atoms, list):
                    raise ValueError('For String method, you should provide at least two structures (initial and final states).')
                if len(self.atoms) < 2:
                    raise ValueError('For String method, you should provide at least two structures (initial and final states).')
                from .algorithm import GSM
                string = GSM(
                    output=self.output,
                    atoms_R=self.atoms[0],
                    atoms_P=self.atoms[1],
                    paras=self.params
                )
                string.run()
                
            elif self.method == 'dimer':
                # OPT-IN batched dimer for a list/Molecules of TS guesses;
                # a single Atoms keeps the unchanged single-structure Dimer.
                if isinstance(self.atoms, (list, Molecules)):
                    self._run_batched_dimer()
                else:
                    from .algorithm import Dimer
                    dimer = Dimer(
                        output=self.output,
                        atoms_init=self.atoms,
                        paras=self.params
                    )
                    dimer.run()

            elif self.method == 'autoneb':
                # AutoNEB: automated multi-step reaction pathway exploration
                if isinstance(self.atoms, Molecules):
                    from .algorithm import AutoNEB
                    autoneb = AutoNEB(
                        output=self.output,
                        atoms_or_molecules=self.atoms,
                        paras=self.params
                    )
                    autoneb.run()
                elif isinstance(self.atoms, list):
                    if len(self.atoms) < 2:
                        raise ValueError('For AutoNEB method, you should provide at least two structures.')
                    from .algorithm import AutoNEB
                    autoneb = AutoNEB(
                        output=self.output,
                        atoms_or_molecules=self.atoms,
                        paras=self.params
                    )
                    autoneb.run()
                else:
                    raise ValueError('For AutoNEB method, you should provide a Molecules object or a list of structures.')

            else:
                raise ValueError(f'Method {self.method} not recognized. Available methods are: prfo, neb, string, dimer, autoneb.')

    # ==================================================================
    # OPT-IN GPU-batched saddle searches (Task: ai-maple-gpu dispatch)
    # ==================================================================
    def _batched_atoms_calc(self):
        """Shared setup: unpack the list/Molecules, resolve the batched calc, and
        wrap the structures in a Molecules whose .calc is the batched calc."""
        from ..dispatcher import resolve_batched_calc
        from maple.function.utility import Molecules
        atoms_list = (self.atoms.multiatoms if isinstance(self.atoms, Molecules)
                      else list(self.atoms))
        if not atoms_list:
            return None, None
        attached = getattr(atoms_list[0], 'calc', None)
        calc = resolve_batched_calc(self.params, atoms_list, attached_calc=attached)
        mols = Molecules(atoms_list)
        mols.calc = calc
        return mols, atoms_list

    def _run_batched_prfo(self):
        """Batched RS-PRFO saddle search over B TS guesses (BatchPRFO). Robust
        config knobs flow through params; single-Atoms PRFO is the oracle."""
        from ..dispatcher import batch_device_str
        from .algorithm import BatchPRFO
        mols, atoms_list = self._batched_atoms_calc()
        if mols is None:
            return
        p = self.params
        bprfo = BatchPRFO(
            output=self.output,
            device=batch_device_str(p),
            recalc=int(p.get('recalc', 4)),
            initial_hessian=str(p.get('initial_hessian', 'identity')),
            ts_hessian_inject=bool(p.get('ts_hessian_inject', False)),
            trust_mode=str(p.get('trust_mode', 'legacy')),
            mode_follow_guard=bool(p.get('mode_follow_guard', False)),
        )
        bprfo.run(mols, pool_queue=p.get('pool_queue'), B_target=p.get('B_target'))

    def _run_batched_dimer(self):
        """Batched dimer saddle search over B TS guesses (BatchDimer). params map
        onto BatchDimerParams; single-Atoms Dimer is the oracle."""
        from ..dispatcher import batch_device_str
        from .algorithm.dimer import BatchDimer
        mols, atoms_list = self._batched_atoms_calc()
        if mols is None:
            return
        bdimer = BatchDimer(
            output=self.output,
            device=batch_device_str(self.params),
            paras=self.params,
        )
        bdimer.run(mols)
