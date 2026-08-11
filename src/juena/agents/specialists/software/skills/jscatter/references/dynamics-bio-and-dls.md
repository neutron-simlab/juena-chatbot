# Dynamics, biomolecules, and DLS

## Model neutron dynamics

First identify the measured observable and domain: intermediate scattering function
`I(q,t)`, dynamic structure factor `S(q,w)`, fixed-window intensity, or another derived
quantity. Confirm the definitions and units of q, time, angular frequency/energy, and
normalization before selecting a function.

- Time-domain building blocks under `js.dynamic` include `transDiff`, `simpleDiffusion`,
  `doubleDiffusion`, `stretchedExp`, `jumpDiffusion`, `methylRotation`,
  `diffusionHarmonicPotential`, `finiteRouse`, `finiteZimm`, and ORZ topology models.
- Frequency-domain building blocks include `elastic_w`, `lorentz_w`, `transDiff_w`,
  `jumpDiff_w`, `stretchedExp_w`, `diffusionHarmonicPotential_w`, `diffusionInSphere_w`,
  `rotDiffusion_w`, and `nSiteJumpDiffusion_w`.
- Convenience models such as `doubleStretchedExp_w`, `threeLorentz_w`, and
  `fixedWindowScanArrhenius` include common composite behavior; verify their exact
  amplitude, background, and resolution conventions before fitting.
- Use `time2frequencyFF`, `frequency2timeFF`, `shiftAndBinning`, `convert_e2w`,
  `mirror_w`, and `dynamicSusceptibility` only after checking transform normalization,
  detailed balance, sampling interval, zero position, and energy/frequency units.

Convolve the physical frequency-domain model with the measured resolution using the
documented `js.dynamic.convolve` or a convenience model that explicitly accepts the
resolution. Do not substitute a guessed Gaussian when resolution data exist. Keep elastic,
quasielastic, inelastic, and flat-background contributions explicit and verify that fitted
linewidths are resolvable.

For NSE or related `dataList` fits, map time to `.X` and q from a per-dataset attribute,
then decide explicitly which amplitudes, diffusion coefficients, elastic fractions, or
stretching exponents are shared. Check q-dependence against the selected mechanism rather
than interpreting each curve independently.

## Analyze biomolecular scattering

Use local PDB/CIF/topology/trajectory inputs from `/inputs` or the configured repository;
network-fetch helpers are unavailable in the sandbox.

- Build a `js.bio.scatteringUniverse` and verify atom names, elements, coordinates,
  periodicity, missing atoms/hydrogens, biological assembly, and position units.
- Define solvent and contrast explicitly. Record D2O fraction, H/D exchange,
  deuteration labeling, hydration-layer density/volume, pH/protonation assumptions, and
  coarse-graining choices.
- Use `nscatIntUniv` for neutron intensity, `xscatIntUniv` for X-ray intensity, or
  `scatIntUniv`/`scatIntUnivYlm` when their general controls are required. Keep coherent
  and incoherent contributions separate in interpretation.
- Use `diffusionTRUnivTensor` or `diffusionTRUnivYlm` for rigid-body translational and
  rotational contributions, with externally justified diffusion inputs when possible.
- Use `intScatFuncYlm`, `intScatFuncPMode`, or `intScatFuncOU` for time-dependent
  scattering only after validating the structural representation and mode amplitudes.
- Use `ANMA`, `vibNM`, `brownianNMdiag`, or related normal-mode classes only with stated
  force/friction weighting, temperature, excluded modes, and convergence checks.

Check absolute-intensity conversions independently. JScatter model output may be per
particle or in another native normalization; concentration and unit conversion must be
shown rather than hidden in a fitted scale.

## Analyze DLS with CONTIN

- Use `js.dls.readZetasizerNano` for supported Zetasizer exports after verifying locale,
  encoding, temperature, viscosity, refractive index, scattering angle, wavelength, and
  correlation-time units.
- Use `js.dls.contin` on the selected correlation data only after deciding the grid,
  time/radius distribution convention, baseline, temperature, and regularization inputs.
- Use `js.dls.contin_display` or the returned fit/distribution attributes for inspection.
- Distinguish intensity-, volume-, number-, relaxation-time-, diffusion-, and
  hydrodynamic-radius-weighted distributions. Do not relabel one as another without the
  required optical and hydrodynamic transformation.
- Treat peak count and peak areas as resolution- and regularization-dependent. Verify the
  reconstructed correlation curve and stability under defensible settings.

## Validate dynamic conclusions

Report resolution treatment, normalization, fitted range, shared parameters, one-sigma
errors when available, correlations, and physical units. Test limiting behavior at
`t=0`, large time, `w=0`, large frequency, and low/high q where applicable. Flag models
whose characteristic rate, linewidth, or mode lies outside the instrument window.
