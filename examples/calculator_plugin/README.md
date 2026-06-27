# Custom calculator plugin — template

This directory is a **copy-and-edit** starting point for plugging your own
machine-learning potential into MAPLE without touching MAPLE's source. It
demonstrates the two most common checkpoint formats and how to load each.

| File | What it is |
|------|------------|
| `my_calculator_plugin.py` | The template. Two ready backends: `mytemplate-jit` (TorchScript `.pt`/`.jpt` via `torch.jit.load`) and `mytemplate-pt` (plain `.pt`/`.model` via `torch.load` + `state_dict`). |
| `build_toy_models.py` | Writes two tiny toy checkpoints so the examples run without a real model. |
| `sp_jit.inp` / `sp_pt.inp` | Single-point inputs that drive each backend through the normal MAPLE CLI. |

## Run the example

```bash
cd examples/calculator_plugin
# make the plugin importable by name + reachable by MAPLE
export PYTHONPATH="$PWD:$PYTHONPATH"

python build_toy_models.py        # writes toy_jit.pt and toy_state.pt
maple sp_jit.inp                  # -> Energy: 0.0039729686 Hartree
maple sp_pt.inp                  # -> same energy via the torch.load path
```

Both inputs print the same energy because they wrap the same toy model loaded
two different ways.

## How MAPLE finds your backend

The header line carries everything:

```text
#model=mytemplate-jit(module=my_calculator_plugin, model_path=toy_jit.pt)
```

- `module=` — the import path of your plugin file/package. MAPLE imports it,
  which runs `@register_calculator` and adds your `MODEL_NAMES` to the registry.
- `model_path=` — the checkpoint MAPLE validates (must exist) and hands to your
  `__init__` via `MODEL_PATH_OPTION`. This is where your `.pt` / `.jpt` /
  `.model` comes in; **your code decides how to load it**.

Alternatively, register the plugin process-wide instead of per-input:

```bash
export MAPLE_CALCULATOR_PLUGINS=my_calculator_plugin
```

## Writing your own

Open `my_calculator_plugin.py` and edit only:

1. The **capability class attributes** (units, PBC/charge support, Hessian modes).
2. The model-load block in `__init__` (the `# TODO` lines).
3. The `_forward` method — map `(atoms)` to a scalar energy in your declared unit.

Everything else (unit conversion, implicit-solvent hook, results bookkeeping,
the analytic-Hessian loop) is handled by `CalcABC` / `_finalize_results` /
`hessian_via_double_autograd`. The full contract is in
[`../../maple/function/calculator/AUTHORING.md`](../../maple/function/calculator/AUTHORING.md)
and the review/rationale in
[`../../maple/function/calculator/CALCULATOR_REVIEW.md`](../../maple/function/calculator/CALCULATOR_REVIEW.md).

> **Unit rule:** declare `MODEL_ENERGY_UNIT` honestly and let `_finalize_results`
> convert the `calculate()` path to Hartree. Do **not** multiply by `EV2HARTREE`
> yourself there. The Hessian path is the one exception and converts internally.
