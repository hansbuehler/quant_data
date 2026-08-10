# Agent Instructions

## Scope and priorities

- This repository is a Python 3.12+ quantitative-data and numerical-research codebase.
- Preserve existing user work and notebook workflows. Do not rewrite unrelated code while making a focused change.
- Treat correctness of measures, densities, sample normalization, tensor shapes, and train/validation separation as higher priority than cosmetic cleanup.

## System and Environment

- Use the existing ``~/vqdata`` repository on Linux and the conda ``%VENV%\vqdata`` environment on Windows.
- We have a linux box "aestas" and a Windows box "sommer" with WSL.
- The linux ``~/python/`` directory on both the linux box and under WSL is mapped onto the Windows ``%HOME%\OneDrive\Python3`` directory. Contents are the same files. Under WSL the same files
  are also visible under a separate ``/mnt/c/Users/hans/OneDrive/Python3``.

- Directories: all three are imported with ``-e .``:
  - ``quant_data`` is a git directory.
  - ``quant_data/dysanos`` is a separate git directory.
  - ``packages/cdxcore`` contains ``cdxcore``.


## Python style

- Use Python 3.12 syntax and built-in generics: `list[int]`, `dict[str, Tensor]`, `X | None`.
- Add type annotations to public functions, configuration fields, and non-obvious return values.
- Use `from __future__ import annotations` in new standalone modules when forward references are useful.
- Use Ruff leniently on changed files. At minimum, check import sorting with `ruff check --select I`; do not introduce broad lint cleanup into a focused change. Do not impose a 100-character target line length.
- Align equal signs across lines where reasonable.
- Add comments before non-obvious blocks where they explain intent, statistical meaning, financial timing, or a numerical safeguard. Do not comment code whose purpose is already clear.
- Follow the modern style in `quantdata/` and `tests/`: four-space indentation, compact but readable expressions, and descriptive snake_case names.
- Do not mechanically reformat legacy research modules as part of an unrelated change. Match the local style where a narrow edit is safer.
- Do not import from random packages. E.g. if ``numpy/np`` is missing import from ``numpy`` not elsewhere. Flag if an import is unknown.
- Prefer explicit keyword-only arguments for configuration-heavy APIs. Use positional parameters only sparingly (e.g. where the parameter is natural from the context).
  Validate data types of parameters using ``verify_inp``.

### Filenames and Unique Labels

- For any string which needs to be a filename validate using ``is_filename`` from ``cdxcore.util``.
- For any string which is unique, but may contain characters which do not allow for a filename, and needs to be converted to a unique filename, use ``label_unique_filename48_8``
  from ``cdxcore.uniquehash``.
- To generate a filename from some ``config`` and a ``label``, use ``named_unique_filename48_8(label, config)`` from ``cdxcore.uniquehash``. However, make sure ``config`` does not
  contain large objects such as dataframes or numpy arrays. For those use the generating config as input.

## Torch

- Derive all project-owned Torch models from ``BaseModel`` from ``quantdata/torch/__init__.py``. This includes project-owned child models rather than only the top-level trainable model.
  - Accept and pass on the same ``env: BaseEnvironment`` to every project-owned child model. This ensures that ``debug``, floating/integer precision, and ``device`` are propagated consistently. To lock a device use ``DeviceLock``.
  - Use the ``BaseEnvironment`` floating type ``fdtype`` and integer type ``idtype`` as default except for data which has a native 
     type which differs (e.g. ``torch.int8`` for flags). Do not covert ``bool``; use ``torch.bool``.
- For feedforward networks use ``FeedForward`` from ``quantdata/torch/ffw.py`` with config ``FeedForwardConfig``.
- Define the model configuration using a `@dataclass(frozen=True, kw_only=True)`.
  The model configuration should contain all model geometry. For example if a feed forward network is used for ``a_y``, include a field
  ``a_y_config : FeedForwardConfig = FeedForwardConfig()``.
- *Versioning*: a model configuration should have:
  - Be decorated by ``@version`` from ``from cdxcore.version import version``, with all dependencies set to any other configurations used elsewhere.
  - A ``user_version`` which is set by the user
  - A ``label`` which must be set by the user. 
  - A ``group`` which is also set by the user. It must be a valid directory structure.
  
  The semantics are as follows:
  - Model checkpoints etc are stored in the cache directory under ``group/``. 
  - The unique model filename body (without exension) is given by the following:
  ```python
  from dataclasses import dataclass

  from cdxcore.uniquehash import label_unique_filename48_8, named_unique_filename48_8
  from cdxcore.version import version

  @version("0.2")
  @dataclass(frozen=True, kw_only=True)
  class ModelConfig:
      label: str
      group: str = ""
      user_version: str = "0.1"
      ...

      @property
      def filename(self) -> str:
          if self.label.startswith("*"):
              return label_unique_filename48_8(self.label[1:])
          return named_unique_filename48_8(self.label, self)
  ```
  - The checkpoint is written with both the code version ``ModelConfig.version.unique_id48`` (``unique_id48`` is a property of ``Version`` from ``cdxcore.version``)
    and the ``user_version``.
    
    If upon deserialization either does not match, the serialization file is out of date and training needs to be cold started.
    Do this check before trying to reconstruct the model as this likely means that the model architecture changed. 
    Print a warning to the user that this happened.

    If the versions match but the model fails to deserialize (e.g. wrong weight geometry) throw an exception. Do not delete the cached checkpoint.
  - Hold a ``FileLock`` from ``filelock`` for the duration of training to prevent another process from training the same logical model and overwriting its checkpoints. Key the lock by the logical checkpoint identity. Do not hold the training lock for read-only loading, valuation, or visualization.

### Training

- Drive training with a ``TrainConfig``. The training config should contain parameters which affect the way training is run but not the
  logical model outcome. A change in a training parameter should allow a warm start from a different training configuration. The user is responsible for deciding whether a changed optimizer, scheduler, precision, gradient-accumulation, or distributed-training configuration is suitable for a warm start. Do not enforce a cold start merely because the ``TrainConfig`` changed. See ``TrainConfig`` in ``quantdata/dysanos/grid/torch_train.py`` as an example.
- Allow compilation through a training-function configuration parameter. Compile performance-critical compatible functions, and retain an eager path for debugging and unsupported operations.
- Allow the user to ``just load`` the model from disk without attempting to train.

### Serialization

- See the naming conventions above.
- Serialize model, optimizer, scheduler, scaler, random-generator, learning-rate, epoch, and other state needed for a near-perfect restart.
- Serialize values by epoch where useful to aid visualization; these may be stored in a separate file.

## Validation and invariants

- Internal consistency - issues which are not caused by user inputs:
  - Use ``assert`` for debug-only internal invariants. Assertions may be disabled with `python -O`; use ``verify`` or an explicit exception for correctness-critical runtime validation. 
  - Assert internal tensor/array shapes near the operation that establishes them. Include expected and actual shapes in the message.
  - Assert generously as long as it does not affect code performance. For example, do not re-sort if an array or datadframe is already expected to be sorted by test
    whether it is sorted (much faster). 
- Validate user/configuration inputs in `__post_init__` with `verify_inp(...)` defined in ``cdxcore.err`` and clear messages that include the invalid value. Wrap the error message
  into a zero parameter ``lambda`` (``verify_inp`` throws ``ValueError``).
- Use `verify(...)` defined in ``cdxcore.err`` or explicit exceptions for runtime/data validation (``verify`` throws ``RuntimeError``),
- For torch models derived from ``BaseModel``:
  ``BaseModel`` provides ``verify``,
  a function which is a NOOP unless ``debug`` is True. Use ``verify`` to validate numeric correctness of torch calculation such as ``isinfinite`` and
  rare logic checks (density being positive). The ``conditions`` argument of ``verify`` is a lambda function which will only be called if ``debug``
  is not ``False``. Similarly, the error message is alsoe a lambda function which will only be formatted if ``debug`` is ``True`` and the conditon
  was not met.
- Check finiteness and sign constraints for probabilities, densities, losses, and other numerically sensitive outputs, in particular before passing to torch and
  when retrieving it. Do that in torch with ``BaseModel.verify`` to avoid expensive checks outside ``debug``.
- Normalize probability/density objects explicitly and verify their normalization.
- Guard divisions, logarithms, variances, and square roots with an appropriate floor or clamp if a violation is possible due to the context. If such violation
  should not be possible (e.g. $u'>0$) then assert/verify instead. Preserve the configured dtype and device.
- Do not silently change statistical conventions. State whether a standard deviation is population or sample based and whether an error uses paths, time steps, or both.

## Configuration

- Parameters of a function/method call often split into two
  * *Scope* parameters which define *what* the function operates on e.g. "ticker", "date", "date range" etc. Usually these are very few. Those are the parameters
    which will usually explicitly show up in data file names or directories, while configs will be shown as the beginning of their ``str(config)`` followed
    by a hash computed using ``named_unique_filename48_8`` from ``cdxcore.uniquehash``.
  * *Config* parameters which define *how* the functions operates e.g. "num_bins", "num_steps" etc.
  * *Processing* parameters are used by the function but which do not affects its numerical outcome e.g. "num_workers", "debug", etc

- Use dataclasses for Configs, usually `@dataclass(frozen=True, kw_only=True)`.
  - Configurations should be universal e.g. exclude *Scope* arguments.
  - With *Scope* and *Config* the returned value of a function should be uniquely defined up to numerical precision.
  - Provide tight ``__str__`` implementations because the standard caching file name is computed using
    ``named_unique_filename48_8(str(config),config)`` using
     ``named_unique_filename48_8`` from ``cdxcore.uniquehash``.

    This function
    - Does not hash members starting with ``_``. Therefore use such protected members for minute numerical parameters (eps) and debugging flags.
    - Hashing large dataframes or NumPy arrays is not sensible. Avoid putting large data objects into configuration dataclasses. Instead, key data with their generating config.
  - Add a ``@version`` decorator with ``from cdxcore.version import version``. Use the ``dependencies=`` keyword to sepcify any
    functions or other configs this config depends on. Examples are all over the code.

## Caching

- Use  ``cdxcore.SubDir.cache`` to transparently cache function results in the data pipeline. Examples are all over the code base.
- Use the ``fmt`` parameter to define the output type to avoid using pickle where not appropriate. Use ``SubDir.POLARS_PARQUET``, ``SubDir.PANDAS_PARQUET``
  or ``SubDir.PYTREE_HDF5`` where suitable. The latter is restricted to pretty native python and numpy.
- Use the ``uid=`` parameter to define a function which computes a unique filename from the function parameters. Do not include parameters which do not
  alter the logical function outcome such as ``num_workers`` or ``Context`` or ``debug``.
  ```python
  from cdxcore.subdir import SubDir
  from cdxcore.uniquehash import named_unique_filename48_8

  _cache = SubDir("~/.cache/", fmt=SubDir.PYTREE_HDF5)

  @_cache.cache(
      "0.1",
      dependencies=[Config],
      uid=lambda ticker, config, num_workers: (
          f"{ticker}/{named_unique_filename48_8(str(config), config)}"
      ),
  )
  def function(ticker: str, config: Config, num_workers: int) -> dict[str, np.ndarray]:
      ...
  ```
- Do **not** cache torch models with this functionality. Use native torch serialization.
- Do **not** write training performance results with this functionality. Use appropriate formarts e.g. WandB.

## Numerical and ML conventions

- Keep NumPy and Torch calculations structurally aligned when both model and visualizer compute the same quantity.
- Construct tensors with the environment/model dtype and device; avoid implicit CPU tensors or dtype promotion.
- Use `torch.no_grad()` for valuation-only calculations and detach diagnostic outputs that must not enter autograd.
- Treat sample axes explicitly. Use names such as `S`, `TAU`, `T`, and `N` consistently with their established meanings, and document reductions over them.
  Validate shapes using ``assert``.
- Distinguish clearly between reference, one-step, bootstrap-adjusted, and compounded path densities. Apply each density exactly once.
- Keep training and validation data separate. Do not use validation data to fit, normalize training quantities, or select parameters unless explicitly requested.
- When minibatches estimate a population quantity, distinguish the estimator's batch observations from the intended effective sample size. Pass global sample counts explicitly when needed.
- Preserve reproducibility: use specific Rng generates (i.e avoid direct use of ``numpy.random``), and seed NumPy and Torch deliberately, serialize RNG state when training can resume, and restore it after setup/warm-up effects. Ensure that if a genuinely new random outcome is desired, this is not accidentially overwritten by setting the seed again.
- When changing objective semantics, density construction, model architecture, or checkpoint interpretation, bump the relevant `@version(...)` dependency so incompatible checkpoints restart rather than load silently.

## Data and financial conventions

- Avoid look-ahead bias. Make the timing and adjustment basis of prices, strikes, reference spots, quotes, trades, and filters explicit.
- Preserve schema and dtype declarations for Polars/dataframe outputs.
- Prefer vectorized NumPy, Polars, Torch, or JAX operations over Python loops for large data.
- Keep units visible in names or documentation, for example `_k`, `_m`, business days, annualized values, or normalized prices.
- Cache expensive external-data and preprocessing operations through the existing `SubDir`/`cdxcore` mechanisms. Include all result-affecting inputs and versioned dependencies in cache identities; exclude only operational arguments such as verbosity and pools.

## Models, outputs, and visualization

- Model outputs are part of the interface. When adding an output, wire it through its tracker, checkpoint serialization, batching/valuation path, and visualizer.
- Keep per-instrument outputs as vectors; aggregate only where a portfolio-level panel explicitly requires it.
- Use the same formula, weighting measure, cost definition, and effective sample count in the model and corresponding visualization. If a plot intentionally uses another measure, label it clearly.
- Follow the established plot colors unless the surrounding plot defines another scheme: training commonly uses red/purple/orange and validation blue/green/light blue.
- Include the model label and group/session in figure titles when multiple sequential measure-change stages can otherwise look identical.
- Legends and axis labels should name the actual statistic being plotted, including cost adjustment and normalization by MC error.
- Write a separate program to visualize all models from a given ``group``. Consider using https://github.com/marimo-team/marimo. ``marimo[recommended]`` is installed.

## Testing and verification

- Use `pytest`; tests live under `tests/` and should be named `test_*.py` with `test_*` functions.
- Add focused regression tests for bug fixes and mathematical tests for changed formulas. Use small deterministic tensors/arrays and `pytest.approx`, `torch.allclose`, or exact shape checks as appropriate.
- Test edge cases relevant to numerical code: zero variance, one observation, empty validation splits, non-finite inputs, device/dtype behavior, and checkpoint compatibility.
- For Torch changes, verify both forward values and gradients when the change affects training.
- Run the narrowest relevant test first, then the containing test module, followed by broader tests when risk warrants it.
- Minimum verification for a modified Python module is a syntax compilation check. Typical commands are:

  ```bash
  python -m py_compile path/to/module.py
  pytest -q tests/test_relevant_module.py
  ruff check --select I path/to/module.py tests/test_relevant_module.py
  ```

- Do not claim notebook-level behavior is verified unless the relevant notebook or an equivalent scripted path was actually executed.

## Documentation and comments

- Public data-access APIs and reusable modules should have useful docstrings; NumPy-style parameter/return sections are preferred for substantial APIs.
- Comments should explain financial timing, statistical meaning, numerical safeguards, or non-obvious design decisions—not restate syntax.
- Keep formulas close to their implementation and name the measure and reduction axes.
- Correct stale comments whenever behavior changes.

## Change discipline

- Keep changes narrow and review the complete diff before handing off.
- Preserve public names and checkpoint/data compatibility unless the task requires a breaking change.
- Do not edit generated notebook output, numbered backups, cached artifacts, or archived implementations as a substitute for changing the canonical source.
- Report which checks ran, and identify any verification that could not be performed.
