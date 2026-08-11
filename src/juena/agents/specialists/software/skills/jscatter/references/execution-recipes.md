# Execution recipes

Adapt these skeletons only after reading the relevant data and physics reference. Verify
all signatures against the installed JScatter version.

## Contents

- Start a headless analysis
- Inspect an input before fitting
- Fit and persist one static dataset
- Plot with GracePlot
- Fit a series with shared and individual parameters
- Fit resolution-smeared SAS data
- Finish safely

## Start a headless analysis

```python
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np
import jscatter as js

js.headless(True)
OUTPUT = Path("/workspace/outputs")
OUTPUT.mkdir(parents=True, exist_ok=True)
print("JScatter", js.__version__)
```

If the printed version is not `1.9.0.1`, inspect `help(...)` or matching source before
using the remaining snippets.

## Inspect an input before fitting

```python
data = js.dA("/inputs/experiment.dat")
print("shape", data.shape)
print("columns", data.columnIndex)
print("attributes", data.attr)
print("finite X/Y", np.isfinite(data.X).all(), np.isfinite(data.Y).all())
if hasattr(data, "eY"):
    print("valid eY", np.isfinite(data.eY).all() and np.all(data.eY > 0))
```

Set column indices only after identifying the file layout:

```python
data.setColumnIndex(ix=0, iy=1, iey=2)
```

## Fit and persist one static dataset

```python
def sphere_model(q, radius, scale, background):
    modeled = js.ff.sphere(q=q, radius=radius)
    modeled.Y = scale * modeled.Y + background
    return modeled

data.setLimit(radius=[0.1, 100.0], scale=[0.0, None], background=[0.0, None])
data.fit(
    model=sphere_model,
    freepar={"radius": 5.0, "scale": 1.0, "background": 0.01},
    fixpar={},
    mapNames={"q": "X"},
    condition=lambda a: np.isfinite(a.Y) & (a.eY > 0),
    method="trf",
)

fit = data.lastfit
fit.savetxt(str(OUTPUT / "sphere-fit.dat"))

fig = Figure()
ax = fig.subplots()
ax.errorbar(data.X, data.Y, yerr=data.eY, fmt=".", label="data")
ax.plot(fit.X, fit.Y, label="fit")
ax.set(xscale="log", yscale="log", xlabel="q / nm$^{-1}$", ylabel="I(q)")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT / "sphere-fit.png", dpi=180)
fig.clear()
```

Replace bounds and units with experiment-specific values. Do not use `[0.0, None]` until
the installed `setLimit` docstring confirms open bounds are accepted; otherwise provide a
finite justified upper bound.

## Plot with GracePlot

Use GracePlot for JScatter-native plots. Headless mode delegates to the preinstalled
`gracebat` executable, so no display server is needed:

```python
plot = js.GracePlot(width=3.4, height=2.4, headless=True)
plot.plot(data, symbol=[1, 0.7, 1], line=[0, 1, 1], legend="measurement")
plot.plot(fit, symbol=0, line=[1, 1, 2], legend="fit")
plot.xaxis(label=r"q / nm\S-1\N", scale="log")
plot.yaxis(label="I(q)", scale="log")
plot.legend()
plot.save(str(OUTPUT / "sphere-fit.agr"))
plot.save(str(OUTPUT / "sphere-fit-grace.png"), size=(3.4, 2.4), dpi=300)
plot.exit()
```

Save `.agr` when later interactive editing or rescaling is valuable. Always call
`plot.exit()` after the final save so queued Grace commands are flushed and the child
process terminates.

## Fit a series with shared and individual parameters

```python
series = js.dL("/inputs/nse-series.dat")

def diffusion(t, amplitude, diffusion_coefficient, wavevector, elastic):
    return amplitude * np.exp(-wavevector**2 * diffusion_coefficient * t) + elastic

series.fit(
    model=diffusion,
    freepar={
        "diffusion_coefficient": 0.1,  # one value shared by all datasets
        "amplitude": [1.0],           # one independent value per dataset
    },
    fixpar={"elastic": 0.0},
    mapNames={"t": "X", "wavevector": "q"},
    condition=lambda a: np.isfinite(a.Y) & (a.eY > 0),
)
series.lastfit.savetxt(str(OUTPUT / "nse-series-fit.dat"))
```

For one parameter per unique metadata value, use JScatter's documented linked-parameter
list syntax ending in the metadata attribute name. Confirm ordering with `np.unique` and
`help(series.fit)`.

## Fit resolution-smeared SAS data

```python
beam = js.sas.prepareBeamProfile(
    "SANS",
    detDist=4000.0,
    collDist=4000.0,
    wavelength=0.5,
    wavespread=0.1,
    collAperture=10.0,
    sampleAperture=5.0,
)

@js.sas.smear(beamProfile=beam)
def smeared_sphere(q, radius, scale, background, beamProfile=None):
    modeled = js.ff.sphere(q=q, radius=radius)
    modeled.Y = scale * modeled.Y + background
    return modeled

sas = js.dA("/inputs/sans.dat")
sas.beamProfile = beam
sas.fit(
    smeared_sphere,
    freepar={"radius": 5.0, "scale": 1.0, "background": 0.01},
    fixpar={},
    mapNames={"q": "X"},
)
```

The numeric instrument values above are placeholders, not defaults. Replace all of them
with verified metadata; q must be in `1/nm` for `js.sas.smear`.

## Finish safely

Before returning, confirm in the same execution that the fit completed, expected
parameters have finite uncertainties when the solver supplies them, and every intended
output write succeeded. Include the reusable Python file itself in `/workspace/outputs`
when the user requests reproducibility. Once the execution tool reports collected
artifacts, do not run another command to inspect those output files.
