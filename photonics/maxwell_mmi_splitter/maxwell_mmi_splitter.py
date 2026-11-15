# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
# ---

# %% [markdown]
# # Maxwell 3D multimode-interference (MMI) 1x2 splitter
# A 1x2 MMI power splitter on a silicon-on-insulator (SOI) platform: a single-mode
# access waveguide feeds a short, wide *multimode* section; the input excites
# several modes that beat against each other, and at the right length the field
# reforms into two identical copies of the input -- a 50/50 split -- caught by two
# output waveguides. This "self-imaging" is the workhorse splitter/combiner of
# silicon photonics.
#
# This is a genuine 3D structure (no effective-index reduction): a 220 nm-thick
# silicon core embedded in SiO2, with real material indices. The simulation axes are
#   * x = propagation along the chip,
#   * y = lateral (across the waveguides, the guide WIDTH direction),
#   * z = vertical (the 220 nm layer-stack / core-thickness direction).
#
# Why a full-wave solver: self-imaging is a coherent interference effect of many
# guided modes; the split ratio and length depend on the modal phase relations, a
# near-field interference pattern with no ray-optics description.
#
# The multimode section's length and the output positions are design inputs; the
# run shows whether the two-fold image lands on the output guides and reads the split.
# The launch is the access guide's own fundamental mode: the engine solves the modes
# of the (y, z) cross-section of the simulation grid when the run starts, with its
# finite-difference mode solver, and drives the one asked for one way along the guide.
# Quasi-TE polarization (dominant E along the guide width, i.e. y).
#
# The guides and the multimode section are drawn as boxes, and the service averages the
# permittivity of every voxel a box face cuts through. The grid is twice as coarse along
# the propagation axis as across it.


# %%
import os
import signal

import numpy as np

from wavesimpro import Box, Medium, Setup, create_mode_source, create_monitor, submit

OUTPUT_DIR = "output"   # figures and saved fields, beside this script

DRY_RUN = False   # True: build the setup and print it, stop before the solve

# %% [markdown]
# ### Parameters (edit these to rescale)

# %%
wavelength = 1.55  # design wavelength in micrometer (μm)
# Grid pitch (dx, dy, dz) in μm; [0.025, 0.025, 0.025] is the isotropic grid. ~9 px across
# the 220 nm core. The facets of the multimode section lie across x, so x is coarsened
# only 2:1.
pixel_size = [0.05, 0.025, 0.025]
dx, dy, dz = pixel_size

n_si = 3.476   # silicon core
n_clad = 1.444  # SiO2 cladding (buried oxide below + oxide over-cladding)

t_core = 0.22  # Si core thickness, vertical z (μm)
W_mmi = 3.0    # multimode-section width, lateral y (μm)
w_wg = 0.45    # single-mode access/output waveguide width (μm)

# The design: the length at which a centred input re-images as two lobes, and where the
# lobes sit. For a centred input the first two-fold image lies at 3/8 of the beat length
# of the two lowest modes, with its lobes a quarter of the section width either side.
L_mmi = 8.0           # multimode-section length (μm)
out_offset = W_mmi / 4  # output guide centres, either side of the axis (μm)

# Transverse window (y lateral, z vertical), core centered in both.
sim_y = W_mmi + 3.0  # lateral domain: MMI width + cladding/boundary margin (μm)
sim_z = t_core + 2.4  # vertical domain: core + oxide margin above/below (μm)
yc = sim_y / 2

# The mode is launched one way by driving a length of guide with the mode's own phase,
# so the backward wave cancels. The drive is centred on x_source and ends 0.5 μm before
# the multimode section.
x_source = 2.0       # centre of the driven length (μm)
drive_length = 3.0   # μm of guide driven
mmi_start = 4.0      # x where the multimode section begins (μm)
mmi_end = mmi_start + L_mmi
y_up, y_lo = yc - out_offset, yc + out_offset

# Absorbing boundary, per axis: padding is added to every face, so size each one by the
# power that reaches it. The guided modes exit through x carrying everything; y and z see
# only radiation off the multimode section.
boundary_widths = [4.0 * wavelength, 1.0 * wavelength, 0.5 * wavelength]

USE_GPU = True   # False routes to the CPU path: slower, far more memory

sim_x = mmi_end + 4.0  # room for the output waveguides
nx = int(round(sim_x / dx))
ny = int(round(sim_y / dy))
nz = int(round(sim_z / dz))
print(f"MMI {W_mmi} x {L_mmi} um from x = {mmi_start} um; outputs at y = {y_up:.2f}, {y_lo:.2f} um")

# %% [markdown]
# ### Build the 3D geometry

# %%
# Silicon boxes in SiO2, 220 nm thick and centred in z. Box positions are in coordinates
# centred on the domain; the access and output guides run on through the x absorbers, so
# the light enters and leaves without meeting an end facet.
size_x, size_y, size_z = nx * dx, ny * dy, nz * dz
reach = boundary_widths[0] + 1.0   # how far the guides continue past the domain
silicon = Medium("Si", n_si)


def guide(name, x_span, y_center, width):
    """A silicon box over ``x_span`` and ``width`` around ``y_center`` (μm from the domain corner)."""
    x0, x1 = x_span
    return Box(n=silicon, name=name, center=((x0 + x1) / 2 - size_x / 2, y_center - size_y / 2, 0.0),
               size=(x1 - x0, width, t_core), subpixel=True)


boxes = [guide("input", (-reach, mmi_start), yc, w_wg),
         guide("multimode", (mmi_start, mmi_end), yc, W_mmi),
         guide("output_1", (mmi_end, size_x + reach), y_up, w_wg),
         guide("output_2", (mmi_end, size_x + reach), y_lo, w_wg)]
mvox = nx * ny * nz / 1e6
print(f"Grid: {nx} x {ny} x {nz} = {mvox:.1f} Mvoxel  (pitch {pixel_size} um)")

# %% [markdown]
# ### Source: the engine solves and launches the access-guide mode

# %%
# The cut is the (y, z) layer where the drive ends, on the access guide before the
# multimode section. At run start the engine solves the full-vector modes of that layer
# of the permittivity the boxes produce and drives the quasi-TE fundamental (E along y,
# across the guide) over ``drive_length`` of guide ending at the cut.
i_drive_end = int(round((x_source + drive_length / 2) / dx))
source = create_mode_source(name="TE0", axis="x", plane_index=i_drive_end, direction=+1,
                            polarization="TE", order=0, drive_length_um=drive_length,
                            engine_solver="fd")
print(f"Mode source: quasi-TE fundamental of the access guide, {drive_length} um driven "
      f"towards +x, ending at x = {i_drive_end * dx:.2f} um")

y = np.arange(ny) * dy

# %% [markdown]
# ### Monitor: return just the x-y cut-plane this figure plots

# %%
# A monitor crops the result to this plane server-side, so only a thin slice is
# transferred instead of the full volume: a slab three voxels thick, centred on the voxel
# at mid core height. Voxel index i on an n-voxel axis sits at (i + 0.5 - n/2) * pitch in
# domain-centred coordinates.
z_mid = nz // 2
z_world = (z_mid + 0.5 - nz / 2) * dz
monitor = create_monitor(
    name="cutplane",
    center_x=0.0, center_y=0.0, center_z=z_world,
    size_x=size_x, size_y=size_y, size_z=3 * dz,
)

# %% [markdown]
# ### Run: submit the setup, read the monitor and the solved mode

# %%
# A setup is one document: the domain and grid, the cladding as background, the boxes,
# the source and the monitor. ``submit`` checks it, saves it on a new project and starts
# the run.
setup = Setup(wavelength, pixel_size, (nx, ny, nz), boundary_widths=boundary_widths,
              background=Medium("SiO2", n_clad), name="maxwell_mmi_splitter")
setup.add(*boxes, source, monitor)

for b in boxes:
    print(f"  {b.name}: x {b.center[0] - b.size[0] / 2 + size_x / 2:.3f} to "
          f"{b.center[0] + b.size[0] / 2 + size_x / 2:.3f} um, y centre {b.center[1] + size_y / 2:.3f} um, "
          f"{b.size[1]} x {b.size[2]} um, sub-pixel averaged")
print(f"  monitor: z = {z_world:+.4f} um, {size_x:.2f} x {size_y:.2f} um, 3 voxels thick")
if DRY_RUN:
    setup.to_properties()
    print("Dry run: setup built, no solve.")
    raise SystemExit

run = submit(setup, use_gpu=USE_GPU, sampling_warnings=False).wait()   # x is coarse on purpose
print(f"Time {run.elapsed:.1f} s; Iterations {run.iterations}; Residual {run.residual_norm}")

u = run.get_monitors()[0]
launched = run.mode_sources()[0]
print(f"Launched mode: neff = {launched['neff']['re']:.4f}")

# Visualization cut-plane: top-down slice at mid core height (the mid z index).
ey = u[1]  # dominant (quasi-TE) component
field_top = ey[:, :, ey.shape[2] // 2]

try:
    signal.signal(signal.SIGINT, signal.default_int_handler)
except (ValueError, TypeError):
    pass

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

os.makedirs(OUTPUT_DIR, exist_ok=True)
intensity = np.abs(field_top) ** 2
intensity /= intensity.max()
extent = [0, sim_x, sim_y, 0]

# %% [markdown]
# ### What the run gives against the 50/50 the design targets

# %%
# The design puts a two-fold image at the end of the section and centres an output guide
# on each lobe. Read the split where the guides begin and report the imbalance in dB --
# the quantity a splitter is actually specified by.
_i_out = int(round((mmi_end + 0.5) / dx))
_out_band = lambda c: np.abs(y - c) < w_wg / 2 + 0.1
_p_up = float(intensity[_i_out, _out_band(y_up)].sum())
_p_lo = float(intensity[_i_out, _out_band(y_lo)].sum())
split_upper = _p_up / (_p_up + _p_lo)
split_imbalance_db = 10 * np.log10(_p_up / _p_lo)
print(f"Split at the output guides: {split_upper:.1%} / {1 - split_upper:.1%} "
      f"against the 50/50 the design targets")
print(f"  imbalance {split_imbalance_db:+.2f} dB")

plt.figure(figsize=(11, 4.5))
plt.imshow(intensity.T, cmap="inferno", extent=extent, norm=LogNorm(vmin=1e-3, vmax=1.0), aspect="equal")
cbar = plt.colorbar(fraction=0.025, pad=0.02)
cbar.ax.set_title(r"$|E_y|^2$")
plt.axvline(mmi_start, color="cyan", lw=0.5, alpha=0.4)
plt.axvline(mmi_end, color="cyan", lw=0.5, alpha=0.4)
plt.xlabel(r"$x~(\mu m)$ -- propagation")
plt.ylabel(r"$y~(\mu m)$ -- lateral")
plt.title(f"3D 1x2 MMI splitter: self-imaging split "
          f"(W={W_mmi} um, L={L_mmi:.1f} um, {split_upper:.1%}/{1-split_upper:.1%} measured, mid-core z)")
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/maxwell_mmi_splitter.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUTPUT_DIR}/maxwell_mmi_splitter.pdf", bbox_inches="tight")
plt.show()
plt.close("all")
