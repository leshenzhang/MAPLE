import sys
import os
import argparse
import traceback

try:
    from maple import __version__ as _VERSION
except Exception:
    try:
        from importlib.metadata import version as _pkg_version
        _VERSION = _pkg_version('maple')
    except Exception:
        _VERSION = '0.1.4'


def _default_output_path(input_file: str) -> str:
    base_name = os.path.splitext(input_file)[0]
    return f"{base_name}.out"


def _log_error_to_output(output_file: str, message: str, exc: Exception | None = None) -> None:
    try:
        with open(output_file, "a") as handle:
            handle.write(f"ERROR: {message.rstrip()}\n")
            if exc is not None:
                traceback.print_exception(type(exc), exc, exc.__traceback__, file=handle)
    except OSError as log_error:
        print(f"Warning: could not write error to output file '{output_file}': {log_error}", file=sys.stderr)


def main():
    """Command-line interface for MAPLE"""

    # Pre-parse to check if 'md' subcommand is used
    # This allows backward compatibility with: maple input.inp
    if len(sys.argv) > 1 and sys.argv[1] == 'md':
        # MD template generator mode
        parser = argparse.ArgumentParser(
            description='MAPLE: MAchine-learning Potential for Landscape Exploration - MD Template Generator',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  maple md nve                 # Generate nve.mdp in current directory
  maple md nvt -o my_nvt.mdp   # Generate with custom filename
  maple md npt -f              # Overwrite existing npt.mdp
            """)
        parser.add_argument('md', help='MD template command')
        parser.add_argument('ensemble', choices=['nve', 'nvt', 'npt'],
                          help='MD ensemble type')
        parser.add_argument('-o', '--output', metavar='FILENAME',
                          help='Output filename (default: {ensemble}.mdp)')
        parser.add_argument('-f', '--force', action='store_true',
                          help='Overwrite existing file without prompting')

        args = parser.parse_args()
        from maple.function.dispatcher.md.md_templates import generate_mdp_template
        try:
            generate_mdp_template(args.ensemble, args.output, args.force)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Normal MAPLE calculation mode
    parser = argparse.ArgumentParser(
        description='MAPLE: MAchine-learning Potential for Landscape Exploration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  maple inp1.inp              # Output to inp1.out
  maple inp1.inp result.out   # Output to result.out
  maple --test 1              # Run test case 1
  maple md nve                # Generate nve.mdp template
  maple md nvt -o my.mdp      # Generate custom named template
        """
    )
    parser.add_argument('input_file', nargs='?', help='Input file path')
    parser.add_argument('output_file', nargs='?', help='Output file path (optional, auto-generated if not provided)')
    parser.add_argument('--test', type=int, choices=range(1, 9),
                        help='Run test case (1-8): 1=LBFGS, 2=NEB, 3=String, 4=Dimer, 5=RFO, 6=IRC, 7=Freq, 8=Scan')
    parser.add_argument('--version', action='version', version=f'%(prog)s {_VERSION}')

    args = parser.parse_args()

    # Original MAPLE calculation logic (default behavior)
    # Test mode
    if args.test:
        # Get the package directory
        import maple
        package_dir = os.path.dirname(os.path.dirname(maple.__file__))
        
        test_cases = {
            1: ('opt/lbfgs', 'inp1.inp', 'LBFGS Optimization'),
            2: ('ts/neb', 'inp1.inp', 'NEB Transition State'),
            3: ('ts/string', 'inp1.inp', 'String Method'),
            4: ('ts/dimer', 'inp1.inp', 'Dimer Method'),
            5: ('opt/rfo', 'inp1.inp', 'RFO Optimization'),
            6: ('irc/gs', 'inp1.inp', 'IRC GS'),
            7: ('freq/mw', 'inp1.inp', 'Frequency MW'),
            8: ('scan', 'C18.inp', 'Scan'),
        }
        
        subdir, filename, description = test_cases[args.test]
        input_file = os.path.join(package_dir, 'examples', subdir, filename)
        print(f"Running test case {args.test}: {description}")
        print(f"Input file: {input_file}")
    else:
        # Normal mode: require input file
        if not args.input_file:
            parser.print_help()
            sys.exit(1)
        input_file = args.input_file

    # Determine output file
    if args.output_file:
        output_file = args.output_file
    else:
        # Auto-generate: inp1.inp -> inp1.out
        output_file = _default_output_path(input_file)

    # Check if input file exists
    if not os.path.exists(input_file):
        message = f"Input file '{input_file}' not found"
        _log_error_to_output(output_file, message)
        print(f"Error: {message}", file=sys.stderr)
        sys.exit(1)

    # Run MAPLE engine
    try:
        from maple.function.engine import engine
        eng = engine()
        eng(input_file, output_file)
        print(f"\nCalculation completed successfully!")
        print(f"Output written to: {output_file}")
    except Exception as e:
        _log_error_to_output(output_file, f"Error during calculation: {e}", e)
        print(f"Error during calculation: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
