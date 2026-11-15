"""Run an example with DRY_RUN set to True, from the example's own folder, without editing it.

    python tools/dry_run.py scattering/maxwell_rough_mirror/maxwell_rough_mirror.py

The example builds its geometry, sources and monitors and writes its preview figures, then
stops before any cloud solve. Only examples that define a module-level DRY_RUN are accepted.
"""
import os
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1]).resolve()
src = path.read_text(encoding="utf-8")
src, n = re.subn(r"^DRY_RUN = .*$", "DRY_RUN = True", src, count=1, flags=re.M)
if n != 1:
    raise SystemExit(f"{path.name} has no DRY_RUN switch")
os.chdir(path.parent)
sys.argv = [str(path)]
exec(compile(src, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
