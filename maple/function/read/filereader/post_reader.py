import os
import re
from typing import List


class PostReader:
    """
    Robust post-processing command reader from .out files.

    Format assumptions (lenient):
      - File contains post-processing commands (C, B, A, D, S) one per line.
      - Empty lines and comments are ignored.
      - Commands are parsed as in InputReader.post_processing_command().

    Supported commands:
      - C i           : Fix atom i
      - B i j         : Fix bond between atoms i and j
      - A i j k       : Fix angle between atoms i, j, k
      - D i j k l     : Fix dihedral between atoms i, j, k, l
      - S ...         : Scan command

    Behavior:
      - The file path must be absolute. If not found, the reader tries 
        case-insensitive lookup in the same directory.
      - Returns a list of command strings (non-empty, stripped).
      - Raises ValueError if the file cannot be found or read.

    Args:
        file_path (str): Absolute path to the .out file.

    Returns:
        List[str]: List of post-processing command strings.
    """

    def __new__(cls, file_path: str) -> List[str]:
        """
        Read and parse a post-processing command file.

        Args:
            file_path (str): Absolute path to the .out file.

        Returns:
            List[str]: List of post-processing commands.

        Raises:
            ValueError: If file not found or invalid format.
        """
        if not os.path.isabs(file_path):
            raise ValueError(f"PostReader requires an absolute path. Got: {file_path}")

        # Try to find the file (case-insensitive if needed)
        resolved_path = cls._resolve_file_path(file_path)
        if not resolved_path:
            raise ValueError(f"PostReader: file not found: {file_path}")

        # Read the file
        try:
            with open(resolved_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            raise ValueError(f"PostReader: failed to read {resolved_path}: {e}")

        # Parse commands
        commands = []
        valid_commands = {'C', 'B', 'A', 'D', 'S'}
        
        for line_num, line in enumerate(lines, start=1):
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                continue
            
            # Parse command
            tokens = stripped.split()
            if not tokens:
                continue
            
            cmd = tokens[0].upper()
            
            # Validate command type
            if cmd not in valid_commands:
                raise ValueError(
                    f"PostReader: invalid command '{cmd}' at line {line_num} in {resolved_path}"
                )
            
            # Basic validation of command format
            if cmd == 'C' and len(tokens) != 2:
                raise ValueError(
                    f"PostReader: C command requires 1 index at line {line_num}: {stripped}"
                )
            elif cmd == 'B' and len(tokens) != 3:
                raise ValueError(
                    f"PostReader: B command requires 2 indices at line {line_num}: {stripped}"
                )
            elif cmd == 'A' and len(tokens) != 4:
                raise ValueError(
                    f"PostReader: A command requires 3 indices at line {line_num}: {stripped}"
                )
            elif cmd == 'D' and len(tokens) != 5:
                raise ValueError(
                    f"PostReader: D command requires 4 indices at line {line_num}: {stripped}"
                )
            elif cmd == 'S' and len(tokens) < 4:
                raise ValueError(
                    f"PostReader: S command requires at least 3 parameters at line {line_num}: {stripped}"
                )
            
            # Validate that indices are integers (except scan step size)
            try:
                if cmd == 'S':
                    # Scan: all but second-to-last should be int, second-to-last is float
                    for idx, token in enumerate(tokens[1:], 1):
                        if idx == len(tokens) - 2:
                            float(token)  # step size
                        else:
                            int(token)    # atom indices and number of steps
                else:
                    # Other commands: all parameters should be integers
                    for token in tokens[1:]:
                        int(token)
            except ValueError:
                raise ValueError(
                    f"PostReader: invalid numeric parameter at line {line_num}: {stripped}"
                )
            
            commands.append(stripped)
        
        if not commands:
            raise ValueError(f"PostReader: no valid commands found in {resolved_path}")
        
        return commands

    @staticmethod
    def _resolve_file_path(file_path: str) -> str:
        """
        Resolve file path with case-insensitive fallback.

        Args:
            file_path (str): Absolute path to check.

        Returns:
            str: Resolved path if found, None otherwise.
        """
        if os.path.isfile(file_path):
            return file_path

        # Try case-insensitive lookup in the same directory
        directory = os.path.dirname(file_path)
        target_name = os.path.basename(file_path).lower()

        if not os.path.isdir(directory):
            return None

        for entry in os.listdir(directory):
            if entry.lower() == target_name:
                candidate = os.path.join(directory, entry)
                if os.path.isfile(candidate):
                    return candidate

        return None