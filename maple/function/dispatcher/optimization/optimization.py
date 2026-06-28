
from ase import Atoms

from ..jobABC import JobABC

from maple.function.timer import timer


class Optimization(JobABC):
    def __init__(self, params: dict, output: str, atoms: Atoms):
        super().__init__(output)
        self.atoms = atoms
        self.commandcontrol = params

    def run(self):
        with timer("Optimization"):
            from maple.function.utility import Molecules
            # OPT-IN GPU-batched optimization for a list/Molecules of structures.
            # Single Atoms is the oracle path below and is left untouched.
            if isinstance(self.atoms, (list, Molecules)):
                return self._run_batched()

            method = str(self.commandcontrol.get('method') or 'lbfgs').lower()
            if method == 'lbfgs':
                from .algorithm import LBFGS
                return LBFGS(self.atoms, output=self.output,
                             paras=self.commandcontrol).run()
            elif method == 'rfo':
                from .algorithm import RFO
                return RFO(self.atoms, output=self.output,
                           paras=self.commandcontrol).run()
            elif method in ('sd', 'sdcg', 'cg'):
                from .algorithm import SDCG
                return SDCG(self.atoms, output=self.output,
                            paras=self.commandcontrol).run()
            else:
                raise NotImplementedError(
                    f"Unknown opt method: {method!r}. "
                    f"Supported: lbfgs, rfo, sd, sdcg, cg.")

    def _run_batched(self):
        """OPT-IN GPU-batched geometry optimization over a list/Molecules.

        Mirrors scan's batched calc-acquisition: build/attach a batched
        calculator, wrap the structures in a Molecules whose ``.calc`` is that
        batched calc, then drive ONE of the validated batched optimizers
        (BatchLBFGS default; sd/sdcg/cg/diis/rfo via params['method']). Each
        batched optimizer reads the per-structure convergence thresholds the
        dispatcher already set on every atom, so it converges to the same minima
        as the single-structure optimizers (the oracle)."""
        from ..dispatcher import resolve_batched_calc, batch_device_str
        from maple.function.utility import Molecules

        params = self.commandcontrol
        atoms_list = (self.atoms.multiatoms if isinstance(self.atoms, Molecules)
                      else list(self.atoms))
        if not atoms_list:
            return None

        method = str(params.get('method') or 'lbfgs').lower()
        attached = getattr(atoms_list[0], 'calc', None)
        calc = resolve_batched_calc(params, atoms_list, attached_calc=attached)
        mols = Molecules(atoms_list)
        mols.calc = calc

        device = batch_device_str(params)
        verbose = int(params.get('verbose', 1))
        maxstep = float(params.get('max_step', 0.2))
        maxiter = int(params.get('max_iter', 256))

        if method == 'lbfgs':
            from .algorithm.blbfgs import BatchLBFGS
            opt = BatchLBFGS(output=self.output, device=device, verbose=verbose,
                             maxstep=maxstep, maxiter=maxiter,
                             memory=int(params.get('memory', 5)),
                             curvature=float(params.get('curvature', 70.0)))
        elif method == 'sd':
            from .algorithm.batch_sd import BatchSD
            opt = BatchSD(output=self.output, device=device, verbose=verbose,
                          max_step=maxstep, max_iter=maxiter)
        elif method in ('sdcg', 'cg'):
            from .algorithm.batch_sdcg import BatchSDCG
            opt = BatchSDCG(output=self.output, device=device, verbose=verbose,
                            max_step=maxstep, max_iter=maxiter, method=method)
        elif method == 'diis':
            from .algorithm.batch_diis import BatchDIIS
            opt = BatchDIIS(output=self.output, device=device, verbose=verbose,
                            maxstep=maxstep, maxiter=maxiter,
                            memory=int(params.get('memory', 6)),
                            min_vectors=int(params.get('diis_min_snapshots', 3)))
        elif method == 'rfo':
            from .algorithm.batch_rfo import BatchRFO
            opt = BatchRFO(output=self.output, device=device, verbose=verbose,
                           max_outer_iter=maxiter)
        else:
            raise NotImplementedError(
                f"Unknown batched opt method: {method!r}. "
                f"Supported: lbfgs, sd, sdcg, cg, diis, rfo.")

        opt.run(mols)
        return mols


# Backward compatibility for the historical misspelling.
Optmization = Optimization
