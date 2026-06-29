from ase import Atoms

from ..jobABC import JobABC

from maple.function.timer import timer

class IRC(JobABC):
    def __init__(self, params: dict, output:str, atoms:Atoms, method:str='gs'):
        super().__init__(output)
        self.atoms = atoms
        self.method = method
        self.output = output
        self.commandcontrol = params

    def run(self):
        # NOTE: every branch RETURNS its algorithm result so a caller / parity
        # harness can consume it. Previously the batched-LQA branch swallowed
        # ``_run_batched_lqa()``'s return value -> ``run()`` yielded ``None`` even
        # though LQABatch routing succeeded, causing a downstream ``float(None)``
        # TypeError when the harness read a result field. (algorithm-completeness
        # fix; default behaviour for the in-place single algorithms is unchanged.)
        with timer("IRC Calculation"):
            if self.method == 'gs':
                from .algorithm import GS
                irc = GS(self.atoms, output=self.output, paras=self.commandcontrol)
                return irc.run()
            elif self.method == 'hpc':
                from .algorithm import HPC
                irc = HPC(self.atoms, output=self.output, paras=self.commandcontrol)
                return irc.run()
            elif self.method == 'eulerpc':
                from .algorithm import EulerPC
                irc = EulerPC(self.atoms, output=self.output, paras=self.commandcontrol)
                return irc.run()
            elif self.method == 'lqa':
                from maple.function.utility import Molecules
                # OPT-IN batched LQA-IRC for a list/Molecules of transition
                # states; a single Atoms keeps the unchanged single LQA.
                if isinstance(self.atoms, (list, Molecules)):
                    return self._run_batched_lqa()
                else:
                    from .algorithm import LQA
                    irc = LQA(self.atoms, output=self.output, paras=self.commandcontrol)
                    return irc.run()
            else:
                raise NotImplementedError(f'IRC method {self.method} not implemented yet.')

    # ==================================================================
    # OPT-IN GPU-batched LQA-IRC (Task: ai-maple-gpu dispatch wiring)
    # ==================================================================
    def _run_batched_lqa(self):
        """Batched LQA-IRC over B transition states via the run_lqa_irc factory
        (-> LQABatch). Single-Atoms LQA is the oracle and is left untouched."""
        from ..dispatcher import resolve_batched_calc, batch_device_str
        from .algorithm.lqa import run_lqa_irc
        from maple.function.utility import Molecules

        atoms_list = (self.atoms.multiatoms if isinstance(self.atoms, Molecules)
                      else list(self.atoms))
        if not atoms_list:
            return None
        attached = getattr(atoms_list[0], 'calc', None)
        calc = resolve_batched_calc(self.commandcontrol, atoms_list,
                                    attached_calc=attached)
        results = run_lqa_irc(atoms_list, output=self.output,
                              paras=self.commandcontrol, calc=calc,
                              device=batch_device_str(self.commandcontrol))
        self._write_batched_lqa(results, atoms_list)
        return results

    def _write_batched_lqa(self, results, atoms_list):
        """Persist a per-structure LQA-IRC summary + merged IRC path .xyz files
        (backward reversed -> TS -> forward) next to the output."""
        import os
        base, _ = os.path.splitext(self.output)
        lines = [
            "\n", "=" * 78 + "\n",
            f"Batched LQA-IRC summary (B = {len(results)})\n",
            "=" * 78 + "\n",
            f"{'idx':>4} {'valid':>6} {'neg_eig':>15} {'E_ts(Ha)':>17} "
            f"{'n_fwd':>6} {'n_bwd':>6}\n",
        ]
        for r in results:
            i = r['index']
            fwd = (r.get('forward') or {}).get('records') or {}
            bwd = (r.get('backward') or {}).get('records') or {}
            fx, fe = fwd.get('x', []), fwd.get('E', [])
            bx, be = bwd.get('x', []), bwd.get('E', [])
            lines.append(
                f"{i:>4} {str(bool(r['valid'])):>6} {r['neg_eigval']:>15.6e} "
                f"{r['E_ts']:>17.8f} {len(fx):>6} {len(bx):>6}\n")
            if not r['valid'] or not (fx or bx):
                continue
            # merged path: backward reversed (ends at TS) + forward without its
            # own TS frame (forward[0] == TS == backward[0]).
            merged_x = list(reversed(bx)) + list(fx[1:])
            merged_e = list(reversed(be)) + list(fe[1:])
            sym = atoms_list[i].get_chemical_symbols()
            with open(f"{base}_irc{i}_full.xyz", "w") as fh:
                for pos, E in zip(merged_x, merged_e):
                    fh.write(f"{len(sym)}\n")
                    fh.write(f"E = {float(E):.10f} Hartree\n")
                    for s, (x, y, z) in zip(sym, pos):
                        fh.write(f"{s:2s} {x: .10f} {y: .10f} {z: .10f}\n")
        lines.append("=" * 78 + "\n")
        with open(self.output, "a") as fh:
            fh.writelines(lines)
