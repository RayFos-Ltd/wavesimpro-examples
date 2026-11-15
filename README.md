# WaveSimPro examples

Advanced examples for [WaveSimPro](https://www.rayfos.com/wavesimpro), the cloud wave-simulation
service: realistic, wavelength-scale devices solved with the full-vector Maxwell or scalar
Helmholtz equations through the [`wavesimpro`](https://pypi.org/project/wavesimpro/) client.
Start with the [`wavesimpro-starter-kit`](https://github.com/RayFos-Ltd/wavesimpro-starter-kit)
if you have not run a simulation yet; the helpers it installs (`wsutils`) are used here too.

Each example is a folder holding a notebook, its script twin and any data it needs. The
`.py` file is the source (Jupytext percent format: it runs as a plain script and opens as a
notebook); the `.ipynb` is generated from it. To edit or add an example, see
[`CONTRIBUTING.md`](CONTRIBUTING.md). Run from inside the example's folder; figures
go to an `output/` folder next to it.

## Setup
```bash
pip install -e .[notebooks]         # from the repository root; installs wavesimpro, wsutils, Jupytext and Jupyter
jupytext --sync free_space_imaging/maxwell_metalens_2d/maxwell_metalens_2d.py   # make the notebook
jupyter lab
```
You need a WaveSimPro account and API key: see the starter kit's README.

## Examples

### Free space & imaging
| Example | Solves | What it shows |
|---|---|---|
| [`maxwell_metalens`](free_space_imaging/maxwell_metalens) | Maxwell 3-D | A flat, wavelength-thin lens (blazed surface-relief kinoform) focuses a plane wave to a diffraction-limited Airy spot; focal cone, spot size and NA. |
| [`maxwell_metalens_nanopillar`](free_space_imaging/maxwell_metalens_nanopillar) | Maxwell 3-D | The same lens as a sub-wavelength nanopillar metasurface, with the pillar-diameter-to-phase library calibrated from periodic unit-cell solves. |
| [`maxwell_metalens_large_aperture`](free_space_imaging/maxwell_metalens_large_aperture) | Maxwell 3-D | A 32 μm amorphous-silicon pillar metalens for the near infrared, solved whole with no periodic boundaries: pillars rasterised with sub-pixel averaging, their phase library calibrated from unit-cell solves, the exit field carried to the focus by the angular spectrum, spot against the diffraction limit and focusing efficiency. Scales to 100 μm apertures. |
| [`maxwell_metalens_2d`](free_space_imaging/maxwell_metalens_2d) | Maxwell 2-D | A 2-D cross-section of the flat lens: the quickest way to see a metalens focus (seconds). |
| [`maxwell_microlens`](free_space_imaging/maxwell_microlens) | Maxwell 2-D | A CMOS sensor microlens a few wavelengths across: focus, angular response over a chief-ray-angle sweep, crosstalk into neighbouring pixels. |
| [`helmholtz_hologram_cgh`](free_space_imaging/helmholtz_hologram_cgh) | Helmholtz 3-D | A Gerchberg–Saxton kinoform reconstructs an image in mid-air by diffraction; rigorous full-wave result against the thin-screen Fourier prediction. |
| [`helmholtz_bessel_vortex`](free_space_imaging/helmholtz_bessel_vortex) | Helmholtz 3-D | A Bessel beam over ~100 wavelengths: non-diffracting core, self-healing past an obstacle, and an optical vortex. |

### Scattering
| Example | Solves | What it shows |
|---|---|---|
| [`helmholtz_random_microspheres`](scattering/helmholtz_random_microspheres) | Helmholtz 3-D | A plane wave through a slab of randomly packed microspheres develops volumetric speckle from multiple coherent scattering. |
| [`helmholtz_random_microspheres_opaque`](scattering/helmholtz_random_microspheres_opaque) | Helmholtz 3-D | The opaque, diffusive regime: speckle contrast and grain size, with a convergence plot. A large run. |
| [`maxwell_buried_ellipsoid`](scattering/maxwell_buried_ellipsoid) | Maxwell 3-D | An ellipsoidal inclusion buried in a dielectric slab, lit obliquely in TE and TM: the field scattered by the inclusion alone (sample minus reference run), side and top cuts, and the angular spectrum of the light scattered back out of the surface. |
| [`maxwell_rough_mirror`](scattering/maxwell_rough_mirror) | Maxwell 3-D | A metal mirror with a random, non-periodic rough surface, lit obliquely in TE and TM: specular spot against diffuse halo in the reflected angular spectrum, flat reflectance against Fresnel, specular share against the Kirchhoff estimate. |

### Photonic crystals
| Example | Solves | What it shows |
|---|---|---|
| [`maxwell_photonic_crystal`](photonic_crystals/maxwell_photonic_crystal) | Maxwell 2-D | A finite square lattice of rods swept in wavelength reveals the photonic bandgap; true transmittance. |
| [`maxwell_pc_waveguide_bend`](photonic_crystals/maxwell_pc_waveguide_bend) | Maxwell 2-D | A line defect guides light around a sharp 90° bend, confined by the bandgap. |

### Image sensors
| Example | Solves | What it shows |
|---|---|---|
| [`maxwell_ipa_light_trapping`](image_sensors/maxwell_ipa_light_trapping) | Maxwell 3-D | Inverted-pyramid light trapping in a CMOS image-sensor surface (Yokogawa et al., *Sci. Rep.* 7:3832): one periodic cell, NIR absorption enhancement against a flat surface. |
| [`maxwell_ipa_pixel`](image_sensors/maxwell_ipa_pixel) | Maxwell 3-D | The full back-illuminated pixel (microlens, inverted pyramids, deep-trench isolation): sensitivity of three surfaces, and optical crosstalk with only the centre pixel lit. |

### Semiconductor inspection
| Example | Solves | What it shows |
|---|---|---|
| [`maxwell_nand_defects`](inspection/maxwell_nand_defects) | Maxwell 3-D | A 3D-NAND stack (tungsten/oxide layers, conical plug holes) with a buried tungsten bridge at four depths, a void and tungsten residue: bright-field difference images through an NA 0.9 objective, and the defect signal against depth. |

### Fibers
| Example | Solves | What it shows |
|---|---|---|
| [`maxwell_fewmode_fiber`](fibers/maxwell_fewmode_fiber) | Maxwell 3-D | A step-index fiber guiding LP01 and LP11 only, launched with its own modes and followed for 350 μm: LP01 alone keeps its profile and its phase advance returns the solved effective index; LP01 and LP11 together beat, the length predicted from the two solved indices against the one measured from the field; a thinner single-mode fiber butt-joined with a lateral offset splits its light between the two modes, the shares read from the field after the joint. |

### Techniques
| Example | Solves | What it shows |
|---|---|---|
| [`helmholtz_pulse2d`](techniques/helmholtz_pulse2d) | Helmholtz 2-D | Time-domain pulse propagation from a frequency sweep: CW solves across a Gaussian spectrum, stacked and inverse-Fourier-transformed into an animation. |

### Integrated photonics
These launch the guides' own modes with the engine's full-vector mode source, which the
service solves from the device's cross-section (as does the few-mode fiber above). The
coupler and the MMI splitter draw their guides as boxes the service sub-pixel averages, on a
grid coarser along the propagation axis than across it.

| Example | Solves | What it shows |
|---|---|---|
| [`maxwell_grating_coupler`](photonics/maxwell_grating_coupler) | Maxwell 3-D | A fibre-to-chip grating coupler on 220 nm SOI, launched with the guide's own fundamental mode, solved and driven by the engine; the device given as a voxel array, as an array refined over the grating, and as sub-pixel boxes, with the beam angle compared. |
| [`maxwell_mmi_splitter`](photonics/maxwell_mmi_splitter) | Maxwell 3-D | A 1×2 multimode-interference splitter launched with the access guide's own mode: section length and output positions as design inputs, the self-imaging split read from the full-wave field. |
| [`maxwell_directional_coupler`](photonics/maxwell_directional_coupler) | Maxwell 3-D | Two side-by-side SOI guides exchange power by supermode beating; laid out for a design coupling length and launched as the engine's two supermodes, whose solved indices predict the coupling length the run then measures from the field. |
