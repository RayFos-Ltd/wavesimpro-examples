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
# # Large-aperture metalens — amorphous-silicon pillars, no periodicity
# A flat lens of amorphous-silicon pillars on a glass substrate, lit through the substrate
# at a near-infrared wavelength. The pillars stand on a sub-wavelength square lattice and
# each one's diameter sets the phase it imprints; the diameters follow the hyperbolic
# focusing phase, so the lens is rotationally symmetric but nowhere periodic. The whole
# aperture is solved in one full-vector Maxwell run with absorbing boundaries on every face.
#
# The pillar library is calibrated first: one pillar in a periodic unit cell, solved for a
# ladder of diameters, gives the transmitted phase and amplitude of each. The lens is built
# from those calibrated pillars and solved as a whole, so pillar-to-pillar coupling, the
# lens edge and the substrate are all in the solve. The field just
# behind the pillars is then carried through the air to the focus with the angular spectrum,
# which is exact in a homogeneous medium: the solved volume only has to hold the lens, and
# its voxel count grows with the aperture squared, not with the focal length.
#
# Results: the pillar library, the lens layout, the field through and behind the pillars,
# the focal cone, the focal spot against the diffraction limit, the measured focal length
# and the focusing efficiency.

# %%
import os
import signal
from time import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from wsutils import apodise
from wavesimpro import simulate, create_monitor, configure

OUTPUT_DIR = "output"   # figures and saved fields, beside this script
os.makedirs(OUTPUT_DIR, exist_ok=True)

DRY_RUN = False   # True: build the lens with a stand-in library, write the setup figure, stop before any solve

# %% [markdown]
# ### Parameters
# Lengths in μm. The x axis is the optical axis: the substrate fills its start and holds the
# source sheet, then come the pillar layer and the exit plane in air. The absorbing boundary
# behind the source continues the glass, so the substrate has no back face: a thin slab
# would be a Fabry-Perot cavity with the lens as one mirror. A 32 μm lens at 40 nm voxels is the
# default; for a 100 μm lens set `aperture = 100.0`, `focal_length = 87.0` and
# `pixel_size = 0.05` (the printed voxel count then tells the memory it needs).

# %%
wavelength = 0.94           # near infrared, where amorphous silicon is transparent
pixel_size = 0.04
n_pillar = 3.6 + 0.01j      # amorphous silicon
n_substrate = 1.45          # fused silica
n_air = 1.0

aperture = 32.0             # lens diameter
focal_length = 28.0         # from the pillar tops
lattice_pitch = 0.6         # square lattice; below the wavelength in the substrate, so no diffraction orders
pillar_height = 0.72
diameters = np.unique(np.round(np.r_[0:0.17:0.02, 0.17:0.2151:0.005, 0.22:0.2501:0.01], 4))
#                             pillar diameters calibrated, 5 nm steps where the phase moves fastest; the
#                             phase completes a cycle by 240 nm and wider pillars hit a transmission dip
probes = 16                 # probe points per voxel and axis for the area-fraction rasterisation
margin = 3.0                # bare substrate around the lens
taper = 2.5                 # Tukey ramp of the source sheet, inside the margin
boundary_width = 2.0        # absorbing boundary on every face, 2 λ: nothing leaves at grazing angles

x_src = 0.5                 # source sheet, inside the substrate
x_inc = (0.9, 1.14)         # two planes in the substrate where the incident amplitude is read
x_pillars0 = 1.5            # top of the substrate, foot of the pillars
x_pillars1 = x_pillars0 + pillar_height
x_exit = x_pillars1 + 1.0   # exit plane: its field is carried to the focus
x_end = x_exit + 0.6

configure(description=f"maxwell_metalens_large_aperture {aperture:.0f} um")   # name of the cloud projects

eps_pillar, eps_substrate, eps_air = n_pillar ** 2, n_substrate ** 2, n_air ** 2
k0 = 2 * np.pi / wavelength
NA = (aperture / 2) / np.hypot(aperture / 2, focal_length)
dl_fwhm = 0.51 * wavelength / NA
transverse = aperture + 2 * margin
nx = int(round(x_end / pixel_size))
ny = nz = int(round(transverse / pixel_size))
n_pitch = int(round(lattice_pitch / pixel_size))
if abs(n_pitch * pixel_size - lattice_pitch) > 1e-9:
    raise ValueError("the lattice pitch must be a whole number of voxels")
x = (np.arange(nx) + 0.5) * pixel_size


def voxel(position):
    """Index of the first voxel whose centre lies at or beyond a position along x."""
    return int(np.floor(position / pixel_size + 1e-9))


ix_src, ix_exit = voxel(x_src), voxel(x_exit)
ix_inc = [voxel(v) for v in x_inc]
ix_layer = np.flatnonzero((x >= x_pillars0) & (x < x_pillars1))   # voxels of the pillar layer
iy_c = ny // 2              # the optical axis: lattice site at the centre of the aperture

nb = int(round(boundary_width / pixel_size))
print(f"Lens: aperture {aperture} um, f {focal_length} um, NA {NA:.2f}; pitch {lattice_pitch} um = {n_pitch} voxels, "
      f"pillars {pillar_height} um tall")
print(f"Domain {x_end:.1f} x {transverse} x {transverse} um = {nx}x{ny}x{nz} voxels ({nx * ny * nz / 1e6:.0f} Mvoxel), "
      f"{wavelength / pixel_size:.0f} voxels per vacuum wavelength")
print(f"With the absorbing boundaries: {(nx + 2 * nb) * (ny + 2 * nb) * (nz + 2 * nb) / 1e6:.0f} Mvoxel")

# %% [markdown]
# ### Pillar rasterisation
# A pillar's edge falls inside voxels. Each voxel of the pillar layer takes the permittivity
# averaged over the fraction of its area inside the pillar's circle, found from a grid of
# probe points, so the diameter varies continuously on the grid instead of stepping by
# whole voxels. The library and the lens are rasterised by the same function.

# %%
cell = np.arange(n_pitch) - n_pitch // 2                # voxel positions in a cell, in voxels, pillar at 0
probe = (np.arange(probes) + 0.5) / probes - 0.5        # probe offsets inside a voxel, in voxels


def pillar_fill(diameter):
    """Fraction of each cell voxel that lies inside the pillar, (n_pitch, n_pitch)."""
    y = (cell[:, None] + probe[None, :]).ravel() * pixel_size
    inside = (y[:, None] ** 2 + y[None, :] ** 2) < (diameter / 2) ** 2
    return inside.reshape(n_pitch, probes, n_pitch, probes).mean(axis=(1, 3))


fills = np.array([pillar_fill(d) for d in diameters])
print(f"{len(diameters)} pillar diameters, 0 to {diameters[-1] * 1e3:.0f} nm, "
      f"{probes}x{probes} probes per voxel")

# %% [markdown]
# ### Library: transmitted phase and amplitude of each diameter
# One pillar in a cell that repeats periodically in y and z, lit by the same source as the
# lens; the transmitted field is averaged over the cell on the exit plane. Phase and
# amplitude are relative to the empty cell, so the first entry means "no pillar". Pillars
# that transmit less than 60% sit on a resonance and are left out of the lens; the
# remaining phases must cover the circle with no gap wider than 0.2 of a cycle, or the
# run stops here rather than solve a lens that cannot focus.

# %%
def unit_cell_medium(fill):
    eps = np.full((nx, n_pitch, n_pitch), eps_air, dtype=np.complex64)
    eps[x < x_pillars0] = eps_substrate
    eps[ix_layer] = eps_air + fill[None] * (eps_pillar - eps_air)
    return eps


def calibrate_library():
    source = np.ones((1, 1, n_pitch, n_pitch), dtype=np.complex64)
    transmitted = []
    t0 = time()
    for k, fill in enumerate(fills):
        u, iterations, residual = simulate(
            permittivity=unit_cell_medium(fill), sources=[(source, (1, ix_src, 0, 0))],
            wavelength=wavelength, pixel_size=pixel_size, boundary_width=boundary_width,
            periodic=(False, True, True), use_gpu=True)
        transmitted.append(u[1, ix_exit].mean())
        t = transmitted[-1] / transmitted[0]
        print(f"  d = {diameters[k] * 1e3:3.0f} nm: {iterations} iterations, "
              f"|t| = {abs(t):.3f}, phase = {np.mod(np.angle(t), 2 * np.pi) / (2 * np.pi):.3f} cycles")
    print(f"Library: {len(fills)} solves in {time() - t0:.0f} s")
    t = np.array(transmitted) / transmitted[0]
    return np.mod(np.angle(t), 2 * np.pi), np.abs(t)


if DRY_RUN:
    phases = np.linspace(0, 2 * np.pi, len(diameters), endpoint=False)   # stand-in: evenly spread phases
    amplitudes = np.ones(len(diameters))
else:
    phases, amplitudes = calibrate_library()
usable = amplitudes >= 0.6
covered = np.sort(phases[usable])
largest_gap = np.max(np.diff(np.append(covered, covered[0] + 2 * np.pi))) / (2 * np.pi)
print(f"{usable.sum()} of {len(diameters)} diameters transmit at least 60%; "
      f"largest gap between their phases {largest_gap:.2f} cycles")
if largest_gap > 0.2:
    raise RuntimeError("the usable pillars leave a gap in the phase circle: change the pitch or the pillar height")

# %% [markdown]
# ### The lens
# Every lattice site inside the aperture gets the usable diameter whose phase is nearest to
# the focusing phase −k₀(√(r² + f²) − f) at that radius. The phase steps between
# neighbouring diameters are the only approximation of the design; the solve below sees the
# lens as it is.

# %%
def build_lens():
    eps = np.full((nx, ny, nz), eps_air, dtype=np.complex64)
    eps[x < x_pillars0] = eps_substrate
    ix0, ix1 = ix_layer[0], ix_layer[-1] + 1
    c = n_pitch // 2
    m = int(aperture / 2 / lattice_pitch)
    sites = iy_c + np.arange(-m, m + 1) * n_pitch        # voxel index of each lattice row and column
    choices = np.flatnonzero(usable)
    placed = []                                           # (y, z, diameter) of every site in the aperture
    for iy in sites:
        for iz in sites:
            r = np.hypot(iy - iy_c, iz - iy_c) * pixel_size
            if r > aperture / 2:
                continue
            target = -k0 * (np.hypot(r, focal_length) - focal_length)
            j = choices[np.argmin(np.abs(np.angle(np.exp(1j * (phases[choices] - target)))))]
            placed.append((iy * pixel_size, iz * pixel_size, diameters[j]))
            if j == 0:
                continue
            eps[ix0:ix1, iy - c:iy - c + n_pitch, iz - c:iz - c + n_pitch] = eps_air + fills[j][None] * (eps_pillar - eps_air)
    placed = np.array(placed)
    print(f"{len(placed)} lattice sites in the aperture, {np.sum(placed[:, 2] > 0)} pillars")
    return eps, placed


permittivity, placed = build_lens()

# %% [markdown]
# ### Figure 1: layout
# The pillar diameter at every lattice site (the rings are the phase wrapping every 2π), a
# zoom on the pillars at the centre, and a side cut through the axis.

# %%
layer = permittivity[ix_layer[len(ix_layer) // 2]].real
extent_yz = [0, transverse, transverse, 0]
fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True,
                                       gridspec_kw={"width_ratios": [1.25, 1, 1.4]})
sc = ax_a.scatter(placed[:, 0], placed[:, 1], c=placed[:, 2] * 1e3, s=3, cmap="viridis", linewidths=0)
ax_a.set_xlim(0, transverse); ax_a.set_ylim(transverse, 0); ax_a.set_aspect("equal")
ax_a.set_xlabel(r"$y~(\mu m)$"); ax_a.set_ylabel(r"$z~(\mu m)$")
ax_a.set_title(f"Pillar diameters, {aperture:.0f} μm aperture")
fig.colorbar(sc, ax=ax_a, fraction=0.046).ax.set_title("nm")
ax_b.imshow(layer.T, cmap="Greys", extent=extent_yz, vmin=1, vmax=n_pillar.real ** 2)
ax_b.set_xlim(transverse / 2 - 3, transverse / 2 + 3); ax_b.set_ylim(transverse / 2 + 3, transverse / 2 - 3)
ax_b.set_xlabel(r"$y~(\mu m)$"); ax_b.set_title("Centre, 6 μm across")
ax_c.imshow(permittivity[:, :, iy_c].real.T, cmap="Greys", extent=[0, x_end, transverse, 0], aspect="auto",
            vmin=1, vmax=n_pillar.real ** 2)
ax_c.axvline(x[ix_src], color="red", ls="--", lw=1)
ax_c.axvline(x[ix_exit], color="tab:blue", ls=":", lw=1)
ax_c.set_ylim(transverse / 2 + 4, transverse / 2 - 4)
ax_c.set_xlabel(r"$x~(\mu m)$, optical axis"); ax_c.set_ylabel(r"$y~(\mu m)$")
ax_c.set_title("Side cut: source (red), substrate, pillars, exit plane (blue)")
fig.savefig(f"{OUTPUT_DIR}/metalens_large_setup.png", dpi=200)

if DRY_RUN:
    print("Dry run: setup figure written, no solve.")
    raise SystemExit

# %% [markdown]
# ### Figure 2: library

# %%
fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
ax.plot(diameters[usable] * 1e3, phases[usable] / (2 * np.pi), "o", color="tab:blue")
ax.plot(diameters[~usable] * 1e3, phases[~usable] / (2 * np.pi), "o", mfc="none", color="tab:blue")
ax.set_ylabel("transmitted phase (cycles)", color="tab:blue")
ax.set_ylim(0, 1)
ax2 = ax.twinx()
ax2.plot(diameters * 1e3, amplitudes, "s--", color="tab:orange")
ax2.axhline(0.6, color="tab:orange", ls=":", lw=0.8)
ax2.set_ylabel("transmitted amplitude", color="tab:orange"); ax2.set_ylim(0, 1.2)
ax.set_title(f"Pillar library: {pillar_height * 1e3:.0f} nm a-Si pillars, {lattice_pitch * 1e3:.0f} nm pitch\n"
             "filled: used in the lens; hollow: below 60% transmission")
ax.set_xlabel("pillar diameter (nm)")
ax.grid(True, alpha=0.3)
fig.savefig(f"{OUTPUT_DIR}/metalens_large_library.png", dpi=200)

# %% [markdown]
# ### Solve
# A y-polarised plane wave, tapered to zero inside the margin, launched in the substrate
# towards the pillars. Three monitors bring back only what is needed: a slab in the glass
# for the incident amplitude, the exit plane, and the x–y cut through the axis.

# %%
source = np.ones((1, 1, ny, nz), dtype=np.complex64)
source, source_position = apodise((source, (1, ix_src, 0, 0)), axes="yz", alpha=2 * taper / transverse)


def world(i, n):
    """Domain-centred coordinate of voxel i on an n-voxel axis, as monitors use it."""
    return (i + 0.5 - n / 2) * pixel_size


monitors = [
    create_monitor(name="incident", center_x=world((ix_inc[0] + ix_inc[1]) / 2, nx), center_y=0.0, center_z=0.0,
                   size_x=(ix_inc[1] - ix_inc[0] + 1) * pixel_size, size_y=transverse, size_z=transverse),
    create_monitor(name="exit", center_x=world(ix_exit, nx), center_y=0.0, center_z=0.0,
                   size_x=pixel_size, size_y=transverse, size_z=transverse),
    create_monitor(name="axial", center_x=0.0, center_y=0.0, center_z=world(iy_c, nz),
                   size_x=x_end, size_y=transverse, size_z=pixel_size),
]

print("\nSolving the lens...")
t0 = time()
(u_inc, u_exit, u_axial), iterations, residual = simulate(
    permittivity=permittivity, sources=[(source, source_position)],
    wavelength=wavelength, pixel_size=pixel_size, boundary_width=boundary_width,
    periodic=(False, False, False), use_gpu=True, monitors=monitors)
print(f"Solved in {time() - t0:.0f} s, {iterations} iterations, residual {residual:.1e}")
try:
    signal.signal(signal.SIGINT, signal.default_int_handler)
except (ValueError, TypeError):
    pass


def plane(u, axis):
    """A one-voxel-thick monitor as a plane: its middle voxel along the thin axis, unless that axis was already dropped."""
    return u if u.ndim == 3 else np.take(u, u.shape[axis] // 2, axis=axis)


u_exit = plane(u_exit, 1)     # (3, ny, nz): the exit plane is thin in x
u_axial = plane(u_axial, 3)   # (3, nx, ny): the axial cut is thin in z
np.savez(f"{OUTPUT_DIR}/metalens_large_fields.npz", exit_plane=u_exit, axial=u_axial, incident=u_inc,
         diameters=diameters, phases=phases, amplitudes=amplitudes)

# %% [markdown]
# ### Incident amplitude
# In the glass the field is the incident wave plus what the lens reflects. Its mean over
# the central aperture on the first and last plane of the slab separates the two, and the
# incident amplitude normalises everything below.

# %%
centre = (np.abs(np.arange(ny) - iy_c) * pixel_size < aperture / 4)
sel = centre[:, None] & centre[None, :]
e = np.array([u_inc[1, 0][sel].mean(), u_inc[1, -1][sel].mean()])
dx = (u_inc.shape[1] - 1) * pixel_size
k_glass = n_substrate * k0
forward, backward = np.linalg.solve([[1, 1], [np.exp(1j * k_glass * dx), np.exp(-1j * k_glass * dx)]], e)
e_inc = abs(forward)
print(f"Incident |E| in the glass {e_inc:.4f}; reflected / incident {abs(backward) / e_inc:.3f}")

# %% [markdown]
# ### From the exit plane to the focus
# The exit-plane field is zero-padded to twice its size, Fourier transformed, and each
# plane wave advanced by exp(i kₓ d) to a distance d; evanescent components are dropped.
# The on-axis intensity against distance locates the focus; the central row of each plane
# builds the focal cone.

# %%
pad = np.pad(u_exit, ((0, 0), (0, ny), (0, nz)))
spectrum = np.fft.fft2(pad, axes=(1, 2))
ky = 2 * np.pi * np.fft.fftfreq(2 * ny, pixel_size)
kz = 2 * np.pi * np.fft.fftfreq(2 * nz, pixel_size)
kx2 = k0 ** 2 - ky[:, None] ** 2 - kz[None, :] ** 2
propagating = kx2 > 0
kx = np.sqrt(np.where(propagating, kx2, 0))


def field_at(distance):
    """The field d behind the exit plane, on the original (unpadded) grid."""
    return np.fft.ifft2(spectrum * np.where(propagating, np.exp(1j * kx * distance), 0), axes=(1, 2))[:, :ny, :nz]


distances = np.linspace(0, 1.5 * focal_length, 300)
on_axis_spectrum = spectrum * np.where(propagating, np.exp(1j * kz * iy_c * pixel_size)[None, None, :], 0)
cone = np.empty((len(distances), ny))
for k, d in enumerate(distances):
    row = np.fft.ifft(np.sum(on_axis_spectrum * np.exp(1j * kx * d), axis=2) / (2 * nz), axis=1)[:, :ny]
    cone[k] = np.sum(np.abs(row) ** 2, axis=0)
on_axis = cone[:, iy_c]
k_focus = int(np.argmax(np.where(distances > 2.0, on_axis, 0)))
d_focus = distances[k_focus]
f_measured = (x[ix_exit] - x_pillars1) + d_focus
print(f"Focus {f_measured:.2f} um from the pillar tops (design {focal_length} um)")

# %% [markdown]
# ### Figure 3: field through the lens and the focal cone

# %%
amp_axial = np.sqrt(np.sum(np.abs(u_axial) ** 2, axis=0)) / e_inc
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True, gridspec_kw={"width_ratios": [1, 2.6]})
im = ax_a.imshow(amp_axial.T, cmap="inferno", extent=[0, x_end, transverse, 0], aspect="auto",
                 norm=colors.LogNorm(vmin=1e-2, vmax=amp_axial.max()))
ax_a.set_xlim(x_src, x_end); ax_a.set_ylim(transverse / 2 + 3, transverse / 2 - 3)
ax_a.axvline(x_pillars0, color="cyan", lw=0.8, ls="--"); ax_a.axvline(x_pillars1, color="cyan", lw=0.8, ls="--")
ax_a.set_xlabel(r"$x~(\mu m)$"); ax_a.set_ylabel(r"$y~(\mu m)$")
ax_a.set_title("Solved field through the centre pillars")
fig.colorbar(im, ax=ax_a, fraction=0.05).ax.set_title(r"$|E|/E_{inc}$")
cone_amp = np.sqrt(cone) / e_inc
im = ax_b.imshow(cone_amp.T, cmap="inferno", aspect="auto",
                 extent=[x[ix_exit] - x_pillars1, x[ix_exit] - x_pillars1 + distances[-1], transverse, 0],
                 norm=colors.LogNorm(vmin=1e-2, vmax=cone_amp.max()))
ax_b.axvline(f_measured, color="white", lw=0.8, ls=":")
ax_b.set_xlabel(r"$x~(\mu m)$ from the pillar tops"); ax_b.set_ylabel(r"$y~(\mu m)$")
ax_b.set_title(f"Focal cone from the exit-plane field: focus at {f_measured:.1f} μm (design {focal_length:.0f} μm)")
fig.colorbar(im, ax=ax_b, fraction=0.03).ax.set_title(r"$|E|/E_{inc}$")
fig.savefig(f"{OUTPUT_DIR}/metalens_large_field.png", dpi=200)

# %% [markdown]
# ### Figure 4: focal spot
# Intensity in the focal plane, its cut against the diffraction-limited width 0.51 λ/NA,
# and the focusing efficiency: the power inside a circle of three FWHM radii around the
# focus, over the power incident on the aperture (|E|² times area, times the refractive
# index on the glass side).

# %%
focal = np.sum(np.abs(field_at(d_focus)) ** 2, axis=0) / e_inc ** 2
cut = focal[:, iy_c] / focal[:, iy_c].max()
above = np.flatnonzero(cut >= 0.5)
fwhm = (above[-1] - above[0] + 1) * pixel_size
yy, zz = np.meshgrid((np.arange(ny) - iy_c) * pixel_size, (np.arange(nz) - iy_c) * pixel_size, indexing="ij")
spot = np.hypot(yy, zz) <= 3 * fwhm
efficiency = focal[spot].sum() * pixel_size ** 2 / (n_substrate * np.pi * (aperture / 2) ** 2)
print(f"Focal spot FWHM {fwhm * 1e3:.0f} nm (diffraction limit {dl_fwhm * 1e3:.0f} nm); "
      f"focusing efficiency {efficiency:.3f}; peak intensity {focal.max():.0f} x incident")

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
im = ax_a.imshow(focal.T, cmap="inferno", extent=[-transverse / 2, transverse / 2, transverse / 2, -transverse / 2])
ax_a.set_xlim(-3, 3); ax_a.set_ylim(3, -3)
ax_a.set_xlabel(r"$y~(\mu m)$"); ax_a.set_ylabel(r"$z~(\mu m)$")
ax_a.set_title(f"Focal plane, {f_measured:.1f} μm from the pillar tops")
fig.colorbar(im, ax=ax_a, fraction=0.046).ax.set_title(r"$|E|^2/E_{inc}^2$")
ax_b.plot((np.arange(ny) - iy_c) * pixel_size, cut, color="tab:orange")
ax_b.axhline(0.5, color="grey", ls=":", lw=0.8)
ax_b.set_xlim(-3, 3); ax_b.set_xlabel(r"$y~(\mu m)$"); ax_b.set_ylabel("normalised intensity")
ax_b.set_title(f"FWHM {fwhm * 1e3:.0f} nm, diffraction limit {dl_fwhm * 1e3:.0f} nm\n"
               f"focusing efficiency {efficiency:.2f}")
ax_b.grid(True, alpha=0.3)
fig.savefig(f"{OUTPUT_DIR}/metalens_large_spot.png", dpi=200)
print(f"Saved {OUTPUT_DIR}/metalens_large_setup.png, _library.png, _field.png, _spot.png")
plt.show()
