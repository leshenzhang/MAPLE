import os
from typing import Union, List

from ase import Atoms 
import ase
import torch

from ..function.read import InputReader
from ..function.dispatcher import Dispatcher
from ..function.utility import Molecules

from maple.function.timer import timer

class engine():
    def __init__(self):
        
        self.output:str = None
        self.gpuid:int = None
        self.model:int = None

        self.jobtype:int = None
        # 1: opt
        # 2: sp
        # 3: scan
        # 4: freq
        # 5: ts

        self.calulator = None

        self.d4 = False
        # DFT-D4 dispersion correction

    @staticmethod
    def _default_output_path(input_file_name: str, output_file_name: str = None) -> str:
        if output_file_name is not None:
            return os.path.abspath(output_file_name)
        return os.path.abspath(os.path.splitext(input_file_name)[0] + ".out")

    @staticmethod
    def _append_error_if_missing(output_path: str, exc: Exception) -> None:
        if not output_path:
            return

        typed_message = f"ERROR: {exc.__class__.__name__}: {exc}\n"
        plain_message = f"ERROR: {exc}\n"
        try:
            existing = ""
            if os.path.exists(output_path):
                with open(output_path, "r", encoding="utf-8", errors="replace") as handle:
                    existing = handle.read()
            if typed_message in existing or plain_message in existing:
                return
            with open(output_path, "a", encoding="utf-8") as handle:
                handle.write(typed_message)
        except Exception:
            # Error logging must never mask the original calculation failure.
            return

    def __call__(self, input_file_name:str,output_file_name:str=None):
        """
            This is the engine of the program.
            Args:
                input_file_name: The path to the input file.
                output_file_name: The path to the output file. (Default: None)
        """
        
        timer.start_total()

        fallback_output = None
        if isinstance(input_file_name, str):
            fallback_output = self._default_output_path(input_file_name, output_file_name)

        try:
            self._input_reader(input_file_name, output_file_name)

            self._mlp_initiator(self.model, self.device)

            if isinstance(self.atoms, Atoms):
                self.atoms.calc = self.calulator
            elif isinstance(self.atoms, Molecules):
                # For Molecules object, set calculator for all atoms in multiatoms
                for atom in self.atoms.multiatoms:
                    atom.calc = self.calulator
            elif isinstance(self.atoms, list):
                # Legacy support for list of Atoms (though now should be Molecules)
                for atom in self.atoms:
                    atom.calc = self.calulator

            # Self.atoms printing
            self._jobtype_dispatcher(
                self.commandcontrol,
                self.jobtype,
                self.atoms,
                self.output,
                extra=self.extra,
            )

            timer.print_summary(self.output)
        except Exception as exc:
            self._append_error_if_missing(self.output or fallback_output, exc)
            raise

    def _input_reader(self, input_file_name:str, output_file_name:str=None) -> Atoms:
        """
            This function reads the input file.

            Args:
                input_file_name: The path to the input file.
                output_file_name: The path to the output file. (Default: Same as input file)
            
            Returns:
                Atoms: ASE Atoms object.
        """
        with timer("Input Reading"):
            reader = InputReader()
            self.atoms = reader(input_file_name, output_file_name)
            self.output = reader.output
            self.device = reader.device
            self.model = reader.model
            self.jobtype = reader.jobtype
            self.d4 = reader.d4
            self.model_options = getattr(reader, "model_options", {}) or {}

            self.extra = {}

            if self.jobtype == 'scan':
                self.extra = {'scan': reader.scan_constraints}
            
            self.commandcontrol = reader.command_control
            
            # Explicit Solvation Treatment
            if self.commandcontrol.get('solv', {}).get('explicit', None) is not None:
                if not isinstance(self.atoms, Atoms):
                    msg = (
                        "Explicit solvation currently supports exactly one structure. "
                        "Split multi-structure/trajectory input before using "
                        "#solv(explicit=...)."
                    )
                    with open(self.output, "a") as handle:
                        handle.write(f"ERROR: {msg}\n")
                    raise ValueError(msg)

                if any(bool(flag) for flag in self.atoms.get_pbc()):
                    msg = (
                        "Explicit solvation is non-periodic; #pbc is not supported "
                        "with #solv(explicit=...)."
                    )
                    with open(self.output, "a") as handle:
                        handle.write(f"ERROR: {msg}\n")
                    raise ValueError(msg)

                from .read import ExplicitSolv
                self.atoms = ExplicitSolv(self.atoms, params=self.commandcontrol.get('solv'), 
                        device=self.device, output=self.output,
                        base_dir=os.path.dirname(reader.input))

    def _mlp_initiator(self, model:str, device: torch.device):
        """
            This function initializes the model.

            Args:
                model: The model to be used.
                device: torch.device
                    The device to run the model on.
        """
        with timer("MLP Initialization"):
            from .calculator import SetClaculator

            implicit_method = self.commandcontrol.get('solv', {}).get('method', None)
            solvent = self.commandcontrol.get('solv', {}).get('implicit', None)
            # Get first atoms object for charge/mult checking
            atoms_for_check = None
            if isinstance(self.atoms, Atoms):
                atoms_for_check = self.atoms
            elif isinstance(self.atoms, (Molecules, list)):
                # For Molecules or list, use first structure
                atoms_list = self.atoms.multiatoms if isinstance(self.atoms, Molecules) else self.atoms
                if atoms_list:
                    atoms_for_check = atoms_list[0]

            setcalculator = SetClaculator(device, model, self.output, atoms=atoms_for_check,
                            d4=self.d4, implicit=implicit_method, solvent=solvent,
                            model_options=self.model_options,
                            solvation_options=self.commandcontrol.get('solv', {}))
            self.calulator = setcalculator.set_calculator()
    
    def _jobtype_dispatcher(self, commandcontrol, jobtype:int, atoms:Union[Atoms, Molecules, List[Atoms]], output:str, extra:dict=None) -> None:
        """
            This function dispatches the job type.

            Args:
                commandcontrol: CommandControl object
                jobtype(int): The type of job to be performed.
                atoms(Union[Atoms, Molecules, List[Atoms]]): The ASE Atoms object, Molecules object, 
                                                            or a list of Atoms objects.
                output(str): The path to the output file.
                method(str): The optimization method to be used. (Default: LBFGS)
                extra(dict): Extra parameters to be passed to the job.
        """
        with timer("Job Dispatching"):
            dispatcher = Dispatcher()
            dispatcher(commandcontrol, jobtype, atoms, output, extra)
    
