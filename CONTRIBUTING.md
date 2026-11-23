# Working on the examples

For whoever edits or adds an example. Readers who only run the examples need the README.

## The pair
Each example is a `.py` in Jupytext percent format plus its executed `.ipynb`. The `.py` is the
source: edit it, not the notebook; it is what gets reviewed, and it runs as a plain script.
After editing, bring the notebook up to date:

```bash
pip install -e .[notebooks]       # once, from the repository root: wsutils, Jupytext, Jupyter
jupytext --sync <name>.py         # updates <name>.ipynb from the edited .py
```

`jupytext --sync` keeps the outputs of every code cell whose text did not change. The notebooks
are executed once per release, so their figures render on GitHub; do not commit a re-executed
notebook outside a release. A script that is not paired yet gets its header with
`jupytext --set-formats ipynb,py:percent <name>.py`.

## Cell conventions
- The first cell is markdown: an H1 title and the example's description.
- Each section is a `# %% [markdown]` cell holding an `### Heading`, followed by a `# %%` code
  cell. Keep prose in markdown cells, not in code comments.
- Images in prose are files in `img/` next to the notebook, referenced by path. Never paste an
  image into a cell: attachments do not survive the `.py` side.

## Adding an example
- One folder per example under its theme, named after the script; data files live beside it.
- After the imports:
  ```python
  NAME = "<name>"                  # the file stem
  OUTPUT_DIR = output_dir(NAME)    # from wsutils: <repo root>/output/<NAME>
  ```
  Name the run on the service after it: `configure(description=NAME)` for `simulate`,
  `Setup(name=NAME)` for `submit`. An example that submits several solves through `submit`
  places them in a Studio folder named `NAME`. Save at least one figure to `OUTPUT_DIR`.
- Helpers come from `wsutils`, the `wavesimpro` client and the scientific Python stack; an
  example imports nothing from outside its own folder.
- Add a row to the README's table for its theme and to its size table, and add the example to
  the feature table where it applies.

## Checks
CI compiles the scripts, checks that each `.py` is a valid Jupytext percent pair and that it
sets `NAME` to its file stem. It never runs a solve.
