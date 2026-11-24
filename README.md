# WaveSimPro examples

Advanced examples for [WaveSimPro](https://www.rayfos.com/wavesimpro), the cloud wave-simulation
service: realistic, wavelength-scale devices solved with the full-vector Maxwell or scalar
Helmholtz equations through the [`wavesimpro`](https://pypi.org/project/wavesimpro/) client.
Start with the [`wavesimpro-starter-kit`](https://github.com/RayFos-Ltd/wavesimpro-starter-kit)
if you have not run a simulation yet; the helpers it installs (`wsutils`) are used here too.

Each example is a folder holding its script, the notebook executed from it and any data it
needs. The `.py` file is the source (Jupytext percent format: it runs as a plain script and
opens as a notebook); the `.ipynb` is that script executed, with its figures. To edit or add
an example, see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Getting started
For an account and API key, see the
[starter kit's README](https://github.com/RayFos-Ltd/wavesimpro-starter-kit).
```bash
pip install -e .[notebooks]   # from the repository root: wavesimpro, wsutils, Jupytext, Jupyter
python -m wavesimpro setup    # once
python free_space_imaging/mx_metalens_2d/mx_metalens_2d.py
```
Each run appears in Studio ([studio.rayfos.com](https://studio.rayfos.com)) under the example's
name, and its figures and saved fields go to `output/<name>/` at the repository root, whichever
folder you run from.

The examples use two call styles: `simulate(...)` takes NumPy arrays (the same call as the
open-source `wavesim` package), and `Setup`/`submit` describes the device with primitives,
mode sources and sub-pixel averaging where the feature needs the service.

## Examples

### Free space & imaging
| Example | Solves | What it shows |
|---|---|---|
| [`mx_metalens`](free_space_imaging/mx_metalens) | Maxwell 3-D | A flat, wavelength-thin lens (blazed surface-relief kinoform) focuses a plane wave to a diffraction-limited Airy spot; focal cone, spot size and NA. |
| [`mx_metalens_large_aperture`](free_space_imaging/mx_metalens_large_aperture) | Maxwell 3-D | A 32 μm amorphous-silicon pillar metalens for the near infrared, solved whole with no periodic boundaries: pillars given as one GDS layer with sub-pixel averaging, their phase library calibrated from unit-cell solves, the exit field carried to the focus by the angular spectrum, spot against the diffraction limit and focusing efficiency. Scales to 100 μm apertures. |
| [`mx_metalens_2d`](free_space_imaging/mx_metalens_2d) | Maxwell 2-D | A 2-D cross-section of the flat lens: the smallest example of a metalens focus. |
| [`mx_microlens`](free_space_imaging/mx_microlens) | Maxwell 2-D | A CMOS sensor microlens a few wavelengths across: focus, angular response over a chief-ray-angle sweep, crosstalk into neighbouring pixels. |
| [`hmhz_hologram_cgh`](free_space_imaging/hmhz_hologram_cgh) | Helmholtz 3-D | A Gerchberg–Saxton kinoform reconstructs an image in mid-air by diffraction; rigorous full-wave result against the thin-screen Fourier prediction. |
| [`hmhz_bessel_vortex`](free_space_imaging/hmhz_bessel_vortex) | Helmholtz 3-D | A Bessel beam over ~100 wavelengths: non-diffracting core, self-healing past an obstacle, and an optical vortex. |

### Scattering
| Example | Solves | What it shows |
|---|---|---|
| [`hmhz_random_microspheres`](scattering/hmhz_random_microspheres) | Helmholtz 3-D | A plane wave through a slab of randomly packed microspheres develops volumetric speckle from multiple coherent scattering. |
| [`hmhz_random_microspheres_opaque`](scattering/hmhz_random_microspheres_opaque) | Helmholtz 3-D | The opaque, diffusive regime: speckle contrast and grain size, with a convergence plot. |
| [`mx_buried_ellipsoid`](scattering/mx_buried_ellipsoid) | Maxwell 3-D | An ellipsoidal inclusion buried in a dielectric slab, lit obliquely in TE and TM: the field scattered by the inclusion alone (sample minus reference run), side and top cuts, and the angular spectrum of the light scattered back out of the surface. |
| [`mx_rough_mirror`](scattering/mx_rough_mirror) | Maxwell 3-D | A metal mirror with a random, non-periodic rough surface, lit obliquely in TE and TM: specular spot against diffuse halo in the reflected angular spectrum, flat reflectance against Fresnel, specular share against the Kirchhoff estimate. |

### Photonic crystals
| Example | Solves | What it shows |
|---|---|---|
| [`mx_photonic_crystal`](photonic_crystals/mx_photonic_crystal) | Maxwell 2-D | A finite square lattice of rods swept in wavelength reveals the photonic bandgap; true transmittance. |
| [`mx_pc_waveguide_bend`](photonic_crystals/mx_pc_waveguide_bend) | Maxwell 2-D | A line defect guides light around a sharp 90° bend, confined by the bandgap. |

### Image sensors
| Example | Solves | What it shows |
|---|---|---|
| [`mx_ipa_light_trapping`](image_sensors/mx_ipa_light_trapping) | Maxwell 3-D | Inverted-pyramid light trapping in a CMOS image-sensor surface (Yokogawa et al., *Sci. Rep.* 7:3832): one periodic cell, NIR absorption enhancement against a flat surface. |
| [`mx_ipa_pixel`](image_sensors/mx_ipa_pixel) | Maxwell 3-D | The full back-illuminated pixel (microlens, inverted pyramids, deep-trench isolation): sensitivity of three surfaces, and optical crosstalk with only the centre pixel lit. |

### Semiconductor inspection
| Example | Solves | What it shows |
|---|---|---|
| [`mx_nand_defects`](inspection/mx_nand_defects) | Maxwell 3-D | A 3D-NAND stack (tungsten/oxide layers, conical plug holes) with a buried tungsten bridge at four depths, a void and tungsten residue: bright-field difference images through an NA 0.9 objective, and the defect signal against depth. |

### Fibers
| Example | Solves | What it shows |
|---|---|---|
| [`mx_fewmode_fiber`](fibers/mx_fewmode_fiber) | Maxwell 3-D | A step-index fiber guiding LP01 and LP11 only, launched with its own modes and followed for 350 μm: LP01 alone keeps its profile and its phase advance returns the solved effective index; LP01 and LP11 together beat, the length predicted from the two solved indices against the one measured from the field; a thinner single-mode fiber butt-joined with a lateral offset splits its light between the two modes, the shares read from the field after the joint. |

### Techniques
| Example | Solves | What it shows |
|---|---|---|
| [`hmhz_pulse2d`](techniques/hmhz_pulse2d) | Helmholtz 2-D | Time-domain pulse propagation from a frequency sweep: CW solves across a Gaussian spectrum, stacked and inverse-Fourier-transformed into an animation. |

### Integrated photonics
These launch the guides' own modes with the engine's full-vector mode source, which the
service solves from the device's cross-section (as does the few-mode fiber above). The
coupler and the MMI splitter draw their guides as boxes on a grid coarser along the
propagation axis than across it.

| Example | Solves | What it shows |
|---|---|---|
| [`mx_grating_coupler`](photonics/mx_grating_coupler) | Maxwell 3-D | A fibre-to-chip grating coupler on 220 nm SOI, launched with the guide's own fundamental mode, solved and driven by the engine; the device given as a voxel array, as an array refined over the grating, and as sub-pixel boxes, with the beam angle compared. |
| [`mx_mmi_splitter`](photonics/mx_mmi_splitter) | Maxwell 3-D | A 1×2 multimode-interference splitter launched with the access guide's own mode: section length and output positions as design inputs, the self-imaging split read from the full-wave field. |
| [`mx_directional_coupler`](photonics/mx_directional_coupler) | Maxwell 3-D | Two side-by-side SOI guides exchange power by supermode beating; laid out for a design coupling length and launched as the engine's two supermodes, whose solved indices predict the coupling length the run then measures from the field. |

## Which example shows what
| Feature | Examples |
|---|---|
| Monitors | `mx_fewmode_fiber`, `mx_metalens`, `mx_metalens_large_aperture`, `mx_ipa_light_trapping`, `mx_ipa_pixel`, `mx_nand_defects`, `mx_directional_coupler`, `mx_grating_coupler`, `mx_mmi_splitter`, `mx_buried_ellipsoid`, `mx_rough_mirror` |
| Per-axis (anisotropic) voxel sizes | `mx_directional_coupler`, `mx_mmi_splitter` |
| Sub-pixel averaging on the service | `mx_grating_coupler`, `mx_buried_ellipsoid`, `mx_fewmode_fiber`, `mx_metalens_large_aperture` |
| Mode sources | `mx_fewmode_fiber`, `mx_directional_coupler`, `mx_grating_coupler`, `mx_mmi_splitter` |
| Primitives (`Box`, `Ellipsoid`) | `mx_directional_coupler`, `mx_grating_coupler`, `mx_mmi_splitter`, `mx_buried_ellipsoid` |
| GDSII layout (`Gds`) | `mx_metalens_large_aperture` |

## Size
Solves is the number of solves an example submits; grid and memory are for its largest solve,
the grid in millions of voxels including the absorbing boundaries, the memory as the service
estimates it.

| Example | Solves | Grid (Mvoxel) | Memory (GB) |
|---|---:|---:|---:|
| `mx_metalens` | 1 | 347 | 18.1 |
| `mx_metalens_large_aperture` | 24 | 216 | 11.3 |
| `mx_metalens_2d` | 1 | 0.52 | 0.03 |
| `mx_microlens` | 4 | 0.10 | 0.005 |
| `hmhz_hologram_cgh` | 1 | 392 | 11.7 |
| `hmhz_bessel_vortex` | 4 | 27.7 | 0.62 |
| `hmhz_random_microspheres` | 1 | 16.6 | 0.37 |
| `hmhz_random_microspheres_opaque` | 1 | 177 | 3.96 |
| `mx_buried_ellipsoid` | 4 | 13.4 | 0.70 |
| `mx_rough_mirror` | 6 | 55.2 | 2.88 |
| `mx_photonic_crystal` | 18 | 1.2 | 0.06 |
| `mx_pc_waveguide_bend` | 1 | 1.0 | 0.05 |
| `mx_ipa_light_trapping` | 4 | 2.6 | 0.14 |
| `mx_ipa_pixel` | 7 | 176 | 13.1 |
| `mx_nand_defects` | 8 | 34.6 | 1.80 |
| `mx_fewmode_fiber` | 5 | 74.3 | 3.87 |
| `hmhz_pulse2d` | 23 | 0.019 | 0.0004 |
| `mx_grating_coupler` | 1 | 334 | 17.4 |
| `mx_mmi_splitter` | 1 | 54.0 | 4.02 |
| `mx_directional_coupler` | 1 | 37.2 | 1.94 |
