# Working on the examples

For whoever edits or adds an example. Readers who only run the notebooks need the README.

## The pair
Each example is a `.py` in Jupytext percent format plus a generated `.ipynb`. The `.py` is the
source: it is what gets reviewed and committed, and it runs as a plain script. The `.ipynb` is
git-ignored and made on demand:

```bash
pip install -e .[notebooks]       # once, from the repository root: wsutils, Jupytext, Jupyter
jupytext --sync <name>.py         # creates or refreshes <name>.ipynb next to it
```

Edit whichever side you like, then run the same `jupytext --sync` again: it updates the older
file from the newer one and keeps the outputs of every code cell whose text did not change, so
a wording fix never costs a re-solve. A file that is not paired yet gets its header with
`jupytext --set-formats ipynb,py:percent <name>.py`.

## Cell conventions
- The first cell is markdown: an H1 title and the example's description.
- Each section is a `# %% [markdown]` cell holding an `### Heading`, followed by a `# %%` code
  cell. Keep prose in markdown cells, not in code comments.
- Images in prose are files in `img/` next to the notebook, referenced by path. Never paste an
  image into a cell: attachments do not survive the `.py` side.

## Adding an example
- One folder per example under its theme, named after the script; data files (GDS, reference
  arrays) live beside it and are opened by bare filename, so the example runs from its folder.
- `OUTPUT_DIR = "output"` after the imports; figures and saved fields go there.
- For an example whose setup takes some understanding (a GDS layout, a mode launch, port
  measurements: things the docstrings alone cannot make clear): a `DRY_RUN = False` switch
  near the top. With it `True` the example builds its medium, sources and monitors, prints
  its audit and writes its preview figures, then stops before the solve, so a reader can see
  what the setup produces before following the physics. `python tools/dry_run.py <path to
  the .py>` runs it that way. A short example that is clear from its text does not need one.
- Helpers come from `wsutils`, the `wavesimpro` client and the scientific Python stack; an
  example imports nothing from outside its own folder.
- Add a row to the README's table for its theme.

## Executing
A solve costs cloud time. Nothing executes notebooks automatically; CI only compiles the
scripts, checks that each `.py` is valid Jupytext, and runs the dry-run switch. Executed
notebooks are produced by hand when an example is published, and are shipped as release assets,
not committed.
