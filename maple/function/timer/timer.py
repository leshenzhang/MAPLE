# -*- coding: utf-8 -*-
"""
Hierarchical Timer for Maple
ORCA-style timing statistics with automatic hierarchy tracking.

Author: Maple Development Team
"""
import time
from contextlib import contextmanager
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TimingNode:
    """
    A single timing node in the hierarchy tree.
    
    Attributes:
        name: Name of this timing section
        level: Depth in the hierarchy (0 = root)
        time: Accumulated time in seconds
        count: Number of times this section was called
        children: List of child timing nodes
    """
    name: str
    level: int
    time: float = 0.0
    count: int = 0
    children: List['TimingNode'] = field(default_factory=list)


class HierarchicalTimer:
    """
    Hierarchical timer with automatic parent-child tracking via call stack.
    
    This timer automatically builds a tree structure of timing information
    by maintaining a stack of active timing contexts. When you enter a timing
    section, it becomes a child of the current top-of-stack section.
    
    The core mechanism:
        1. Maintain a stack representing current position in the tree
        2. start() pushes a new node onto the stack
        3. stop() pops the node and accumulates time
        4. Tree structure emerges naturally from stack operations
    
    Usage:
        # Method 1: Context manager (recommended)
        with timer("Section Name"):
            do_work()
        
        # Method 2: Manual (use only when necessary)
        timer.start("Section Name")
        do_work()
        timer.stop("Section Name")
        
        # At program end
        timer.print_summary()
    """
    
    def __init__(self):
        """Initialize the timer with an empty tree."""
        # Root node represents the entire program
        self.root = TimingNode("Total", level=0)
        
        # Stack tracks current position in tree (always starts with root)
        self.stack: List[TimingNode] = [self.root]
        
        # Map from full path to node (for quick lookup)
        self.flat_timings: Dict[str, TimingNode] = {}
        
        # Map from name to start time (for currently running timers)
        self._start_times: Dict[str, float] = {}
        
        # Global start time for the entire program (wall time)
        self._global_start: Optional[float] = None
        
        # CPU start time
        self._cpu_start: Optional[float] = None
        
        # Datetime for start and end
        self._datetime_start: Optional[datetime] = None
        self._datetime_end: Optional[datetime] = None
    
    def reset(self):
        """Reset all timing data. Useful for timing multiple independent runs."""
        self.__init__()
    
    def start_total(self):
        """
        Start the global timer for the entire program.
        Call this at the beginning of your main function.
        Records wall time, CPU time, and datetime.
        """
        self._global_start = time.perf_counter()
        self._cpu_start = time.process_time()
        self._datetime_start = datetime.now()
    
    @contextmanager
    def __call__(self, name: str):
        """
        Context manager interface for timing a code block.
        
        This is the recommended way to use the timer. It automatically
        handles start/stop pairing and works correctly even if exceptions occur.
        
        Args:
            name: Name of the timing section
            
        Yields:
            None
            
        Example:
            with timer("Energy Calculation"):
                energy = atoms.get_potential_energy()
        """
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)
    
    def start(self, name: str):
        """
        Start timing a section.
        
        This method:
        1. Records the start time
        2. Looks for existing child node with this name under current parent
        3. If not found, creates a new child node
        4. Pushes the child node onto the stack (becomes new "current" node)
        
        Args:
            name: Name of the timing section
            
        Raises:
            RuntimeError: If a timer with this name is already running
        """
        if name in self._start_times:
            raise RuntimeError(f"Timer '{name}' already started (missing stop call?)")
        
        # Record start time
        self._start_times[name] = time.perf_counter()
        
        # Get current parent (top of stack)
        parent = self.stack[-1]
        
        # Look for existing child with this name
        child = None
        for c in parent.children:
            if c.name == name:
                child = c
                break
        
        # Create new child if not found
        if child is None:
            child = TimingNode(name, level=parent.level + 1)
            parent.children.append(child)
            
            # Add to flat index for debugging/analysis
            full_name = self._get_full_path(child)
            self.flat_timings[full_name] = child
        
        # Push child onto stack (becomes new "current" node)
        self.stack.append(child)
    
    def stop(self, name: str):
        """
        Stop timing a section and accumulate the elapsed time.
        
        This method:
        1. Calculates elapsed time since start
        2. Pops the node from the stack
        3. Accumulates time and increments call count
        4. Stack top automatically returns to parent
        
        Args:
            name: Name of the timing section (must match corresponding start call)
            
        Raises:
            RuntimeError: If the timer was not started or name doesn't match
        """
        if name not in self._start_times:
            raise RuntimeError(f"Timer '{name}' was not started")
        
        # Calculate elapsed time
        elapsed = time.perf_counter() - self._start_times.pop(name)
        
        # Pop node from stack
        node = self.stack.pop()
        
        # Verify name matches (catches mismatched start/stop pairs)
        if node.name != name:
            raise RuntimeError(
                f"Timer mismatch: expected '{node.name}', got '{name}'. "
                f"Check for missing stop() calls in nested timers."
            )
        
        # Accumulate time and count
        node.time += elapsed
        node.count += 1
    
    def _get_full_path(self, node: TimingNode) -> str:
        """
        Get the full path from root to node (for flat index).
        
        Args:
            node: The node to get path for
            
        Returns:
            Path string like "Optimization > Energy > Forward Pass"
        """
        names = []
        # Walk up the stack to build path
        for item in self.stack[1:]:  # Skip root
            names.append(item.name)
        names.append(node.name)
        return " > ".join(names)
    
    def print_summary(self, output_file: str):
        """
        Print ORCA-style timing summary to file only (no stdout output).
        
        The output shows:
        - Start and end datetime
        - Tree structure with indentation showing hierarchy
        - Time and percentage for each section
        - Call count for sections called multiple times
        - Total wall time and CPU time
        - Formatted run time in days/hours/minutes/seconds
        
        Args:
            output_file: Path to output file (required). Summary will be appended.
        
        Example output:
            Program started: 2025-12-02 10:30:45
            
            ======================================================================
                                  TIMING SUMMARY
            ======================================================================
            Optimization ................................. 2.450 s  ( 83.6 %)
              Energy & Forces (×50) ...................... 2.000 s  ( 81.6 %)
                Energy (×50) ............................. 1.000 s  ( 50.0 %)
                Forces (×50) ............................. 1.000 s  ( 50.0 %)
              Line Search (×50) .......................... 0.450 s  ( 18.4 %)
            ======================================================================
            Total wall time: 2.930 s
            Total CPU time: 2.850 s
            ======================================================================
            
            Program ended: 2025-12-02 10:30:48
            TOTAL RUN TIME: 0 days 0 hours 0 minutes 2 seconds 930 msec
        """
        # Record end time
        self._datetime_end = datetime.now()
        
        # Calculate total times
        if self._global_start is not None:
            total_wall_time = time.perf_counter() - self._global_start
            self.root.time = total_wall_time
        else:
            total_wall_time = self.root.time
        
        if self._cpu_start is not None:
            total_cpu_time = time.process_time() - self._cpu_start
        else:
            total_cpu_time = 0.0
        
        # Build output lines
        lines = []
        
        # Start datetime
        if self._datetime_start:
            lines.append(f"\n\n\nProgram started: {self._datetime_start.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("")
        
        lines.append("=" * 70)
        lines.append("                      TIMING SUMMARY")
        lines.append("=" * 70)
        
        # Recursively print tree structure (pass None as parent_time for root level)
        for child in self.root.children:
            self._print_node(child, lines, total_wall_time, parent_time=total_wall_time)
        
        lines.append("=" * 70)
        lines.append(f"Total wall time: {total_wall_time:.3f} s")
        lines.append(f"Total CPU time: {total_cpu_time:.3f} s")
        lines.append("=" * 70)
        lines.append("")
        
        # End datetime
        if self._datetime_end:
            lines.append(f"Program ended: {self._datetime_end.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Format total run time in ORCA style
        total_ms = int(total_wall_time * 1000)
        days = total_ms // (24 * 3600 * 1000)
        total_ms %= (24 * 3600 * 1000)
        hours = total_ms // (3600 * 1000)
        total_ms %= (3600 * 1000)
        minutes = total_ms // (60 * 1000)
        total_ms %= (60 * 1000)
        seconds = total_ms // 1000
        msec = total_ms % 1000
        
        lines.append(f"TOTAL RUN TIME: {days} days {hours} hours {minutes} minutes {seconds} seconds {msec} msec")
        lines.append("")
        
        # Write to file only (no stdout output)
        output = "\n".join(lines)
        with open(output_file, 'a') as f:
            f.write(output)
    
    def _print_node(self, node: TimingNode, lines: List[str], total_time: float, parent_time: float):
        """
        Recursively print a node and its children.
        
        Percentage is calculated relative to parent node's time, not total time.
        This makes it easier to see how time is distributed within each section.
        
        Same-level nodes have their time columns aligned.
        Different levels have different alignment positions (shifted right).
        
        Args:
            node: Node to print
            lines: List of output lines to append to
            total_time: Total execution time (for root level percentage)
            parent_time: Parent node's time (for calculating relative percentage)
        """
        # Calculate percentage relative to parent
        percentage = (node.time / parent_time * 100) if parent_time > 0 else 0
        
        # Format indentation based on level
        indent = "  " * (node.level - 1)
        
        # Add call count if called multiple times
        if node.count > 1:
            name_part = f"{indent}{node.name} (×{node.count})"
        else:
            name_part = f"{indent}{node.name}"
        
        # Format time and percentage with fixed widths
        time_part = f"{node.time:7.3f} s"
        pct_part = f"({percentage:6.1f} %)"
        
        # Each level has its own time column position
        # Level 1: column 52
        # Level 2: column 54 (shifted right by 2)
        # Level 3: column 56 (shifted right by 4)
        # etc.
        base_column = 52
        time_column = base_column + (node.level - 1) * 2
        
        # Calculate dots needed to reach the time column for this level
        if len(name_part) < time_column:
            dots = "." * (time_column - len(name_part))
        else:
            dots = " "  # Fallback if name is too long
        
        # Combine into aligned output line with dots
        line = f"{name_part}{dots} {time_part}  {pct_part}"
        lines.append(line)
        
        # Recursively print children (pass this node's time as parent_time)
        for child in node.children:
            self._print_node(child, lines, total_time, parent_time=node.time)
    
    def get_dict(self) -> Dict:
        """
        Get timing data as a dictionary (useful for debugging or export).
        
        Returns:
            Dictionary with total time and hierarchical breakdown
            
        Example:
            {
                "total_time": 10.5,
                "breakdown": {
                    "time": 10.5,
                    "count": 1,
                    "children": {
                        "Optimization": {
                            "time": 8.2,
                            "count": 1,
                            "children": {...}
                        }
                    }
                }
            }
        """
        return {
            "total_time": self.root.time,
            "breakdown": self._node_to_dict(self.root)
        }
    
    def _node_to_dict(self, node: TimingNode) -> Dict:
        """
        Recursively convert a node to dictionary format.
        
        Args:
            node: Node to convert
            
        Returns:
            Dictionary representation of the node
        """
        result = {
            "time": node.time,
            "count": node.count,
        }
        if node.children:
            result["children"] = {
                child.name: self._node_to_dict(child)
                for child in node.children
            }
        return result


# =============================================================================
# Global singleton instance
# =============================================================================
# This is the key to making the timer work across multiple files without
# passing timer objects around. All modules import the same instance.
timer = HierarchicalTimer()


# =============================================================================
# Optional: Decorator for timing entire functions
# =============================================================================
def timed(name: Optional[str] = None):
    """
    Decorator to automatically time a function.
    
    This is syntactic sugar for wrapping the entire function body
    in a timing context. Use this when you want to time a complete
    function without modifying its internal code.
    
    Args:
        name: Name for the timing section. If None, uses function name.
        
    Returns:
        Decorated function
        
    Example:
        @timed("Energy Calculation")
        def compute_energy(self, atoms):
            # ... function body
            return energy

        # Equivalent to:
        def compute_energy(self, atoms):
            with timer("Energy Calculation"):
                # ... function body
                return energy
    """
    def decorator(func):
        nonlocal name
        if name is None:
            name = func.__name__
        
        def wrapper(*args, **kwargs):
            with timer(name):
                return func(*args, **kwargs)
        
        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator