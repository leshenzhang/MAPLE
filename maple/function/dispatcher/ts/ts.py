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
                # OPT-IN batched GSM: a list of [R,P] reactions (each element a
                # pair/list/Molecules) routes to GSMBatch; a flat [R,P] keeps the
                # unchanged single GSM oracle.
                if (isinstance(self.atoms, list) and self.atoms
                        and isinstance(self.atoms[0], (list, tuple, Molecules))):
                    return self._run_batched_gsm()
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
                # OPT-IN batched AutoNEB: a list of reactions (each element a
                # pair/list/Molecules) routes to AutoNEBBatch; a single Molecules
                # or flat structure list keeps the unchanged single AutoNEB oracle.
                if (isinstance(self.atoms, list) and self.atoms
                        and isinstance(self.atoms[0], (list, tuple, Molecules))):
                    return self._run_batched_autoneb()
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

            elif self.method == 'geodesic':
                # Training-free geodesic TS-guess: build the geodesic path R->P in
                # the Morse-scaled interatomic-distance metric (optionally FIRE-relax
                # + climb on the MLIP) and take the highest-energy node as the TS
                # guess. Output feeds straight into BatchPRFO. Accepts a Molecules
                # ([R, P]) or a list [R, P].
                self._run_geodesic_guess()

            else:
                raise ValueError(f'Method {self.method} not recognized. Available methods are: prfo, neb, string, dimer, autoneb, geodesic.')

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

    def _run_geodesic_guess(self):
        """Training-free geodesic TS-guess generation. A single reaction
        (Molecules([R,P]) / list [R,P]) writes one TS guess; for a batch of
        reactions call GeodesicTSGuess.run_multiband directly. The guess is a
        valid input to the 'prfo' method (BatchPRFO) for exact refinement."""
        from .algorithm import GeodesicTSGuess
        from maple.function.utility import Molecules
        if isinstance(self.atoms, Molecules):
            mol_in = self.atoms
        elif isinstance(self.atoms, list):
            if len(self.atoms) < 2:
                raise ValueError('For geodesic method, provide reactant + product (>=2 structures).')
            mol_in = Molecules(self.atoms)
        else:
            raise ValueError('For geodesic method, provide a Molecules object or a list [reactant, product].')
        # Attach a batched calc if one is not already carried (enables MLIP HEI
        # picking / FIRE relax; without it a geometric midpoint node is returned).
        if getattr(mol_in, 'calc', None) is None:
            try:
                from ..dispatcher import resolve_batched_calc
                attached = getattr(mol_in.multiatoms[0], 'calc', None)
                mol_in.calc = resolve_batched_calc(self.params, mol_in.multiatoms,
                                                   attached_calc=attached)
            except Exception:
                pass
        geo = GeodesicTSGuess(output=self.output, atoms_or_molecules=mol_in,
                              paras=self.params)
        geo.run()

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
            # OPT-IN VRAM-adaptive pool sizing (default OFF -> fixed B_target).
            auto_batch=bool(p.get('auto_batch', False)),
            auto_batch_cap=int(p.get('auto_batch_cap', 256)),
            vram_safety=float(p.get('vram_safety', 0.8)),
        )
        bprfo.run(mols, pool_queue=p.get('pool_queue'), B_target=p.get('B_target'),
                  auto_batch=p.get('auto_batch'))

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

    def _run_batched_gsm(self):
        """Batched GSM over B reactions (run_gsm -> GSMBatch). Each reaction is a
        [R,P] pair / Molecules; the single-reaction GSM stays the oracle."""
        from ..dispatcher import resolve_batched_calc, batch_device_str
        from .algorithm.string import run_gsm
        reactions = list(self.atoms)
        if not reactions:
            return None
        flat = [a for rxn in reactions
                for a in (rxn.multiatoms if isinstance(rxn, Molecules) else rxn)]
        attached = getattr(flat[0], 'calc', None) if flat else None
        calc = resolve_batched_calc(self.params, flat, attached_calc=attached)
        return run_gsm(reactions, output=self.output, paras=self.params,
                       calc=calc, device=batch_device_str(self.params))

    def _run_batched_autoneb(self):
        """Batched AutoNEB over B reactions (AutoNEBBatch). Each reaction is a
        [R,P] pair / Molecules; the single-reaction AutoNEB stays the oracle."""
        from ..dispatcher import resolve_batched_calc, batch_device_str
        from .algorithm.autoneb import AutoNEBBatch
        reactions = list(self.atoms)
        if not reactions:
            return None
        flat = [a for rxn in reactions
                for a in (rxn.multiatoms if isinstance(rxn, Molecules) else rxn)]
        attached = getattr(flat[0], 'calc', None) if flat else None
        calc = resolve_batched_calc(self.params, flat, attached_calc=attached)
        return AutoNEBBatch(self.output, reactions, calc=calc,
                            paras=self.params).run()
