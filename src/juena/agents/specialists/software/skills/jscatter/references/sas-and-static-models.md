# SAS reduction and static scattering models

Use this reference to reduce SANS/SAXS data, calculate contrast, construct static
models, include polydispersity and interactions, and fit instrumental resolution. Verify
signatures against the installed JScatter version when adapting the examples.

## Contents

- Keep reduction, physics, and resolution separate
- Establish the unit contract
- Prepare one-dimensional SAS data
- Reduce a detector image deliberately
- Calculate SLD and contrast
- Understand form-factor output and absolute intensity
- Select the minimum justified model
- Build a homogeneous-particle model
- Add a structure factor
- Average over polydispersity
- Build core-shell and multi-shell models
- Fit contrast or concentration series
- Construct and apply a beam profile
- Fit a resolution-smeared model
- Handle explicit resolution and merged data
- Use desmearing only when justified
- Treat lattices and oriented scattering separately
- Validate the physical result
- Avoid common mistakes

## Keep reduction, physics, and resolution separate

Write the analysis in three visible layers:

```text
detector counts
  -> calibrated, corrected I(q) with uncertainty
  -> scale * particle scattering P(q) * interaction S(q) + background
  -> resolution-smeared prediction at measured q
```

Before changing measured values, record which corrections the facility pipeline has
already applied. Check detector sensitivity, dead time, dark current, empty beam,
container, solvent, transmission, exposure, sample thickness, solid angle, absolute
scale, masks, detector geometry, and radial or sector averaging. Never apply a
correction twice or infer a missing instrument value from a plausible default.

`js.sas.transmissionCorrection(...)` modifies a `dataArray` **in place**. It is designed
for specific raw SAXSpace/Ganesha-style metadata or for explicitly supplied `dark`,
`emptybeam`, `exposure`, `transmission`, and `Izero` values. Read its docstring and
match its normalization equation to the experiment before using it. Preserve an
uncorrected copy and propagate uncertainties through any later subtraction or scaling.

## Establish the unit contract

JScatter SAS APIs do not use one universal unit convention. Write units next to every
quantity and use this table for the APIs covered here:

| Context | Coordinate/length units |
| --- | --- |
| `js.ff.sphere`, examples in this reference | `q` in `nm^-1`, dimensions in `nm`, SLD/contrast in `nm^-2` |
| `js.sf.PercusYevick` with `molarity` | `q` in `nm^-1`, radius in `nm`, molarity in `mol/L` |
| `js.sas.smear` and explicit resolution sigma | `q` and Gaussian sigma in `nm^-1` |
| `prepareBeamProfile("SANS", ...)` | distances/apertures/pixel widths in `mm`, wavelength in `nm`, `wavespread` as relative FWHM |
| `js.sas.sasImage(...)` | detector distance and pixel size in `m`, wavelength in angstrom, center as `[height, width]` pixels |

The `sasImage` wavelength and `prepareBeamProfile` wavelength therefore differ by a
factor of ten for the same beam. For example, `5.0` angstrom in `sasImage` is `0.5` nm
in `prepareBeamProfile`. Convert deliberately; do not copy the number unchanged.

Some form- and structure-factor functions describe their units relative to a supplied
length instead of fixing nanometres. Read the selected function's docstring before
mixing it with the examples below.

## Prepare one-dimensional SAS data

Use the `/skills/jscatter/references/data-and-fitting.md` reference for the complete `dataArray` contract. For a
typical reduced dataset, create or verify rows in `(columns, points)` orientation:

```python
import jscatter as js
import numpy as np

# q: nm^-1, intensity and sigma_I: cm^-1
data = js.dA(
    np.vstack((q, intensity, sigma_I)),
    XYeYeX=[0, 1, 2],
)
data.columnname = "q; intensity; sigma_I"
data.q_unit = "nm^-1"
data.intensity_unit = "cm^-1"
```

Before fitting, confirm:

1. `data.X`, `data.Y`, and mapped `data.eY` mean q, intensity, and one-sigma
   uncertainty rather than simply being the first three file columns.
2. q values are finite, positive where the model requires it, sorted, and expressed in
   the model's unit.
3. uncertainties are finite and positive, and include propagated reduction errors where
   available.
4. masks, transmission, thickness, concentration, contrast, and absolute-normalization
   provenance are retained as metadata or in the reproducible script.
5. overlapping detector configurations agree within uncertainty before combination.

Keep detector configurations separate when they need different resolution profiles or
backgrounds. A `dataList` can fit them simultaneously without first merging their q
values.

## Reduce a detector image deliberately

Construct `sasImage` only from a calibrated detector file or an array with explicit
geometry:

```python
image = js.sas.sasImage(
    "/inputs/sample.tif",
    detector_distance=4.0,       # m
    center=[512.4, 510.8],       # [height, width] pixels
    pixel_size=[75e-6, 75e-6],   # m
    wavelength=5.0,              # angstrom
    maskbelow=0,
)

image.showattr()
print(image.center, image.detector_distance, image.pixel_size, image.wavelength)
```

File metadata can supply these values, but inspect the resulting attributes rather than
assuming they were parsed correctly. The equivalent explicit setters are
`setDetectorDistance`, `setPlaneCenter`, `setPixelSize`, `setWavelength`, and
`setPlaneOrientation`.

Mask the beamstop, detector gaps, dead or saturated pixels, shadows, and other invalid
regions before averaging. Inspect the two-dimensional pattern and an azimuthal average
for anisotropy. A radial average destroys directional information and is unsuitable when
the sample is oriented or anisotropic.

For isotropic, raw counting data:

```python
radial = image.radialAverage(
    number=250,
    kind="log",
    calcError="poisson",
    units="sr",
)
radial = radial.where(lambda a: (a.X >= q_min) & (a.X <= q_max))
```

`units="sr"` produces counts per microsteradian and accounts for detector-distance
changes. `radialAverage` also applies the flat-detector projection correction. Use
`calcError="poisson"` only for unprocessed photon/neutron counts from a counting
detector. After background subtraction, normalization, interpolation, or other
processing, propagate the existing uncertainty instead; `calcError="std"` is bin
variation, not automatically the uncertainty of the corrected mean.

## Calculate SLD and contrast

Calculate X-ray or neutron scattering-length densities with an explicit composition,
density, isotope state, and temperature:

```python
silica_nsld = js.formel.scatteringLengthDensityCalc(
    "1Si1O2",
    density=2.65,   # g/cm^3
    T=293.15,       # K
    mode="ncohsld",
)

h2o_nsld = js.formel.scatteringLengthDensityCalc(
    "1H2O1",
    density=0.998,
    T=293.15,
    mode="ncohsld",
)

d2o_nsld = js.formel.scatteringLengthDensityCalc(
    "1D2O1",
    density=1.1056,
    T=293.15,
    mode="ncohsld",
)

contrast = silica_nsld - h2o_nsld  # coherent neutron contrast, nm^-2
```

Chemical formulas require a number after every element: use `Si1O2`, `H2O1`, and
`D2O1`. `D` denotes deuterium. For a mixture, pass a list whose leading numbers are
molar amounts when `units="mol"` or mass amounts for other `units` values:

```python
buffer_nsld = js.formel.scatteringLengthDensityCalc(
    ["55000D2O1", "137Na1Cl1", "50Na3P1O4"],
    T=293.15,
    units="mol",
    mode="ncohsld",
)
```

With `density=None`, JScatter attempts its water-solution density calculation and needs
an H2O/D2O component. Supply a measured or justified density for other materials.
Record H/D exchange, labile hydrogens, deuteration fraction, solvent mixing rule,
temperature, and density source. Never use X-ray SLD as neutron SLD or vice versa.

Useful `mode` results include `xsld`, `edensity`, `ncohsld`, `incsld`, and `num`.
`xsld`, coherent neutron SLD, and incoherent SLD are returned in `nm^-2`; electron
density is in electrons per `nm^3`; number density is in `nm^-3`.

## Understand form-factor output and absolute intensity

Many `js.ff` functions return a `dataArray`, not just a vector. For a homogeneous
sphere:

```python
particle = js.ff.sphere(q=q, radius=12.0, contrast=2.5e-4)
print(particle.shape, particle.columnname)
# rows: q, Iq, fa

single_particle_I = particle.Y
amplitude = particle[2]
print(particle.I0, particle.fa0)
```

For `js.ff.sphere`, `.Y` is the squared single-particle amplitude
`contrast**2 * volume**2 * shape(q)**2`; row 2 is the signed amplitude. With q and
radius in the units above, the result has units `nm^2`. Do not square `.Y` again.
Scale and offset only `.Y`: arithmetic on the whole returned `dataArray` would also change
`q` and the errors (see `/skills/jscatter/references/data-and-fitting.md`).

Measured absolute intensity also needs a number density and any experimentally justified
normalization:

```python
def dilute_spheres(q, radius, contrast, number_density, background):
    ff = js.ff.sphere(q=q, radius=radius, contrast=contrast)
    return number_density * ff.Y + background
```

Here `number_density` in `nm^-3` times sphere scattering in `nm^2` yields `nm^-1`.
Convert to the measured intensity unit explicitly; `1 nm^-1 = 1e7 cm^-1`. That
conversion applies only when the model really uses this absolute normalization. Many
JScatter models have different normalization contracts, so inspect their return
documentation instead of applying a universal scale.

Keep empirical scale, particle number density, coherent contrast, incoherent scattering,
and additive background separate. A free scale can hide incorrect concentration,
contrast, units, or absolute calibration.

## Select the minimum justified model

Choose a model from morphology and the measured q range, not solely from a visually
similar curve:

- Use `guinier`, `genGuinier`, `guinierPorod`, `powerLaw`, or `beaucage` for a stated
  asymptotic or empirical regime. Restrict a Guinier fit to a defensible low-q range.
- Use `sphere`, `ellipsoid`, `triaxialEllipsoid`, `cylinder`, `disc`, `cuboid`, `prism`,
  or `superball` for compact homogeneous particles.
- Use `sphereCoreShell`, `multiShellSphere`, `multiShellCylinder`,
  `multiShellEllipsoid`, `multiShellBicelle`, `multilayer`, or
  `multilamellarVesicles` only when the data and contrast support internal structure.
- Use polymer/soft-matter functions such as `gaussianChain`, `wormlikeChain`,
  `ringPolymer`, `teubnerStrey`, or the ORZ models only with their stated chain,
  solvent, and normalization assumptions.
- Use `cloudScattering` or oriented-cloud functions when coordinates or orientation
  distributions are part of the model.

Check the exact callable and its return columns with `help(js.ff.<name>)`. A parameter
called `radius`, `R`, `Rc`, or `Rs` is not interchangeable by assumption, and some
functions expect SLDs while others expect a precomputed contrast.

## Build a homogeneous-particle model

Start with a model that exposes all dimensional factors:

```python
def sphere_model(q, radius, contrast, number_density, background):
    particle = js.ff.sphere(
        q=q,
        radius=radius,
        contrast=contrast,
    )
    return number_density * particle.Y + background

initial = sphere_model(
    q=data.X,
    radius=10.0,
    contrast=contrast,
    number_density=1e-5,
    background=0.0,
)
assert initial.shape == data.X.shape
assert np.all(np.isfinite(initial))
```

For relative-intensity data, replace number density with a clearly named empirical
`scale`. Do not call it concentration unless it has the corresponding unit and
normalization.

Use physically justified starting values and limits. For example, keep radius and
number density positive, but do not impose a narrow size limit merely to force an
optimizer to converge. Apply the fitting workflow and residual checks from
`/skills/jscatter/references/data-and-fitting.md`.

## Add a structure factor

For monodisperse hard spheres under the Percus-Yevick approximation:

```python
def interacting_spheres(
    q, radius, contrast, number_density, hard_sphere_radius, volume_fraction, background
):
    particle = js.ff.sphere(q=q, radius=radius, contrast=contrast)
    interaction = js.sf.PercusYevick(
        q=q,
        R=hard_sphere_radius,
        eta=volume_fraction,
    )
    return number_density * particle.Y * interaction.Y + background
```

`PercusYevick` returns a dimensionless structure factor. Supply either `eta` or
`molarity`; when `molarity` is supplied it is in `mol/L` and fixes q/radius to
`nm^-1`/`nm`. Check whether the hard-sphere interaction radius should equal the form
factor radius for the physical system; a stabilizing layer can make them different.
If `eta` and `number_density` describe the same effective hard spheres, verify
`eta = 4*pi*hard_sphere_radius**3*number_density/3` rather than fitting them as
independent physical quantities without justification.

The product `P(q) * S(q)` assumes a factorization/decoupling approximation. Size
polydispersity, orientation, shape-dependent interactions, aggregation, multiple
species, and concentration-dependent particle structure can invalidate it. State the
assumption before fitting. Other `js.sf` functions, including sticky/adhesive hard
spheres, RMSA, Yukawa, polyelectrolyte, fractal, and lamellar-stack models, each impose
different potentials or geometries; inspect their docstrings and parameter limits.

## Average over polydispersity

Use `js.formel.parDistributedAverage` or its shortcut `js.formel.pDA` when a parameter
has a distribution and the wrapped function returns a `dataArray`:

```python
distributed = js.formel.pDA(
    js.ff.sphere,
    2.0,                 # standard deviation in nm
    "radius",
    type="lognorm",
    nGauss=40,
    q=q,
    radius=12.0,         # distribution mean in nm
    contrast=contrast,
)

print(distributed.radius_mean, distributed.radius_std)
print(distributed.pdf)
```

JScatter averages every returned row except `.X`; for `sphere` this averages both the
single-particle intensity and amplitude over the distribution. Use `.Y` for the
polydisperse scattering. Averaging intensity represents a number-weighted population of
independent particles; squaring the averaged amplitude would describe a different
coherent problem.

For `norm`, `lognorm`, and `gamma`/`schulz`, `sig` is the intended standard deviation
corresponding to the supplied mean. The other distributions differ:

- `truncnorm` uses `sig` as the scale of the underlying normal and `a`, `b` as cutoffs
  measured in that scale; truncation changes the resulting mean and standard deviation,
  which JScatter records as `<parameter>_mean` and `<parameter>_std`;
- `lorentz`/`cauchy` uses `sig` as the Cauchy scale or HWHM and records
  `<parameter>_FWHM = 2*sig`; its mathematical mean and standard deviation are undefined;
- `uniform` uses `sig` as the full interval width.

Read the exact docstring and inspect the returned `.pdf` before fitting `sig`. Avoid a
normal radius distribution that assigns meaningful probability below zero; use a
positive distribution or justified truncation. Do not apply `pDA` to a model that
already integrates the same dispersity.

When combining polydispersity and interactions, decide whether the structure factor is
evaluated at the mean size, averaged consistently, or coupled through a more complete
closure. Do not multiply a polydisperse form factor by a monodisperse structure factor
without disclosing that approximation.

## Build core-shell and multi-shell models

Use explicit SLD profiles when contrast variation or internal particle structure is
resolved:

```python
core_shell = js.ff.multiShellSphere(
    q=q,
    shellthickness=[8.0, 2.0],       # inner core radius, then shell thickness; nm
    shellSLD=[core_sld, shell_sld],  # nm^-2
    solventSLD=solvent_sld,          # nm^-2
)

print(core_shell.shellradii)
print(core_shell.contrastprofile)
```

`multiShellSphere` treats the first thickness as the innermost sphere radius and later
values as successive shell thicknesses. It returns rows `[q, Iq, Fa]`; `.Y` is the
single-particle scattering cross-section in `nm^2` under the units used here. Multiply
by a number concentration to obtain an absolute macroscopic intensity.

`sphereCoreShell(q, Rc, Rs, bc, bs, solventSLD=0)` is a convenience wrapper in which
`Rc` and `Rs` are core and outer radii and `Rs > Rc`. In 1.9.0.1 its docstring calls
`bc` and `bs` contrasts, but the implementation forwards them as `shellSLD` values to
`multiShellSphere`, which then subtracts `solventSLD`. Avoid that ambiguity:

- pass core and shell **contrasts** as `bc`/`bs` only with `solventSLD=0`; or
- use `multiShellSphere` directly for absolute core, shell, and solvent SLDs.

Inspect `.contrastprofile` and `.shellcontrast` before fitting either representation.

Adding shells introduces correlated radii, thicknesses, SLDs, and scale. Use contrast
series or independent composition/density information to identify them; otherwise fix
unresolvable SLDs or simplify the model rather than reporting unstable parameters.

## Fit contrast or concentration series

Keep datasets separate in a `dataList` when solvent SLD, concentration, background, or
resolution differs. Attach the physical values to each dataset:

```python
series = js.dL()
for dataset, solvent_sld, concentration, beam_profile in experiments:
    dataset.solventSLD = solvent_sld
    dataset.number_density = concentration
    dataset.beamProfile = beam_profile
    series.append(dataset)
```

Write the model so geometry can be shared while dataset attributes supply contrast,
concentration, or resolution. In `dataList.fit`, a scalar starting value is shared, a
one-element list creates one value per dataset, and `[start, "attribute_name"]` links
one value per unique metadata group; verify the group ordering before fitting. A single
`dataArray.fit` keeps only the first element of such a list.

Share a radius or shell thickness only when sample preparation and physics justify it.
Allow per-dataset backgrounds and profiles when instruments or configurations differ.
For a concentration series, check whether concentration changes `S(q)`, aggregation,
or particle structure instead of assuming only a multiplicative scale changes.

## Construct and apply a beam profile

Use verified instrument values to prepare a SANS-like resolution profile:

```python
beam = js.sas.prepareBeamProfile(
    "SANS",
    collDist=8000.0,       # mm
    collAperture=15.0,     # mm
    detDist=8000.0,        # mm
    sampleAperture=5.0,    # mm
    wavelength=0.5,        # nm = 5 angstrom
    wavespread=0.10,       # relative FWHM Delta-lambda/lambda
    dpixelWidth=8.0,       # mm
    dringwidth=1.0,        # detector-bin width in pixels
)
```

The values above demonstrate units only; replace every number with instrument metadata.
Do not rely on JScatter's defaults as measurement facts. A measured line-collimation
profile, a trapezoidal profile, a single Gaussian width, or per-q explicit Gaussian
widths can also be passed to `prepareBeamProfile`; read its docstring for the selected
profile type.

Smear a model with the profile, preferably using the decorator so JScatter evaluates an
extended q range at the measured edges:

```python
@js.sas.smear(beamProfile=beam)
def smeared_spheres(q, radius, contrast, number_density, background):
    particle = js.ff.sphere(q=q, radius=radius, contrast=contrast)
    particle.Y = number_density * particle.Y + background
    return particle
```

The model coordinate name must begin with `q` or `Q`, and q must be in `nm^-1`.
The wrapper returns the smeared prediction in `.Y` while retaining useful unsmeared and
resolution information in additional rows/attributes. Compare smeared and unsmeared
curves to see whether fitted size/dispersity is resolution-sensitive.

## Fit a resolution-smeared model

Evaluate the wrapped model before fitting and then use the ordinary `dataArray` fit
contract:

```python
trial = smeared_spheres(
    q=data.X,
    radius=12.0,
    contrast=contrast,
    number_density=1e-5,
    background=0.0,
)
assert trial.Y.shape == data.Y.shape
assert np.all(np.isfinite(trial.Y))

data.setLimit(radius=(0.1, 1000.0), number_density=(0.0, np.inf))
data.fit(
    model=smeared_spheres,
    freepar={"radius": 12.0, "number_density": 1e-5, "background": 0.0},
    fixpar={"contrast": contrast},
    mapNames={"q": "X"},
    method="trf",
)
```

For simultaneous datasets with distinct resolution, include `beamProfile=None` in the
model signature and attach a prepared profile to each dataset:

```python
@js.sas.smear(beamProfile=default_beam)
def multi_resolution_model(
    q, radius, contrast, number_density, background, beamProfile=None
):
    particle = js.ff.sphere(q=q, radius=radius, contrast=contrast)
    particle.Y = number_density * particle.Y + background
    return particle

data_near.beamProfile = near_beam
data_far.beamProfile = far_beam
series = js.dL([data_near, data_far])
```

During `dataList.fit`, matching dataset attributes are passed to the model. A model call
with `beamProfile=None` uses the profile supplied by the decorator; it does **not**
bypass smearing. For a SANS profile, a resolution component such as `detDist=None` or
`collDist=None` causes the wrapper to return the unsmeared model. Inspect the evaluated
model for every dataset instead of assuming all profiles were applied.

## Handle explicit resolution and merged data

For a dataset containing one Gaussian q-sigma per point, identify the physical width
row explicitly:

```python
# Rows: q, intensity, sigma_I, sigma_q; q and sigma_q both in nm^-1
explicit_beam = js.sas.prepareBeamProfile(data, explicit=3)
```

Every q point needs a corresponding width. q values must be unique, especially in an
overlap between detector distances; otherwise the wrong width can be selected. Missing
widths are interpolated and can mix configurations unexpectedly.

Best practice is to retain non-merged detector configurations as separate datasets with
their own beam profiles, then fit them simultaneously. If a merged product is the only
input, plot q versus width, inspect configuration transitions, remove duplicate q values
with a documented rule, and test whether the result changes when overlap points are
excluded.

## Use desmearing only when justified

Prefer fitting a smeared model. `js.sas.desmear(Ios, beamProfile, ...)` applies an
iterative Lake-style correction to measured data, which can amplify noise and depends
on low- and high-q extrapolation at detector edges. Use it only when the user explicitly
needs desmeared data and the extrapolation is physically defensible.

If desmearing is required:

1. preserve the measured data unchanged;
2. inspect the beam profile and edge extrapolation functions;
3. report iteration/stopping choices;
4. compare reconvolved desmeared data with the measurement;
5. show noise and edge sensitivity across reasonable settings;
6. never feed a desmeared curve into a smeared model again.

## Treat lattices and oriented scattering separately

For ordered phases, construct a lattice with verified cell parameters and symmetry, then
use the applicable `js.sf` lattice function such as `latticeStructureFactor`,
`radial3DLSF`, or `orientedLatticeStructureFactor`. Validate predicted peak positions,
multiplicities, orientation, finite-size broadening, and instrument resolution before
interpreting intensities.

Do not reduce anisotropic detector data to a radial average and then fit an oriented
model. Use q-vector coordinates and the relevant two- or three-dimensional form factor,
orientation distribution, and detector geometry. Powder averaging, orientational
averaging, mosaicity, and instrumental smearing are distinct operations.

## Validate the physical result

In addition to the fit checks in `/skills/jscatter/references/data-and-fitting.md`, require:

- residuals without low-q aggregation trends, misplaced minima/maxima, or systematic
  high-q background errors;
- correct limiting behavior, such as a justified Guinier range and plausible Porod or
  other asymptotic slope;
- dimensions supported by features within the measured q range rather than extrapolated
  beyond it;
- SLD and contrast consistent with composition, density, isotope state, and solvent;
- scale/number density consistent with concentration and absolute calibration where
  those are claimed;
- volume fraction and interaction parameters within the selected structure factor's
  physical domain;
- no unresolved tradeoff among scale, contrast, number density, size distribution,
  background, and resolution width;
- stable parameters across reasonable starting values, q ranges, and resolution
  assumptions;
- simultaneous-series trends consistent with contrast and concentration physics;
- a visible comparison of smeared and unsmeared predictions when resolution matters.

Save the reduction/model script, prepared data, `lastfit`, and diagnostic plots under
`/workspace/outputs`. Report the JScatter version, all unit conversions, correction
ledger, model components, SLD inputs, distribution, structure-factor approximation,
resolution source, fit range, solver, uncertainties, correlations, and remaining
physical ambiguity.

## Avoid common mistakes

- Do not confuse `(points, columns)` tables with JScatter's `(columns, points)` layout.
- Do not assign Poisson errors to already corrected intensities.
- Do not copy a wavelength number between `sasImage` and `prepareBeamProfile`; their
  wavelength units are angstrom and nanometres, respectively.
- Do not assume every form factor returns normalized `P(q)`; inspect return rows and
  units.
- Do not square a returned intensity or use an amplitude row as intensity.
- Do not hide missing absolute normalization or wrong contrast inside a free scale.
- Do not multiply `P(q)` and `S(q)` without stating the factorization assumptions.
- Do not average amplitude when independent-particle intensity should be averaged.
- Do not add polydispersity twice.
- Do not use merged q values with missing, duplicated, or mismatched resolution widths.
- Do not desmear data merely to make a plot look sharper.
- Do not use a radial average for anisotropic scattering.
- Do not report shell SLDs, dispersity, or interaction parameters when correlations show
  that the available q range or contrast cannot identify them.
