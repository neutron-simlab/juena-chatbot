# `dataArray`, `dataList`, and fitting

Use this reference to create, inspect, prepare, fit, and save JScatter datasets. Verify
signatures against the installed version when adapting the examples.

## Contents

- Understand the `dataArray` layout
- Create one dataset from NumPy
- Read one dataset from a file
- Inspect before assigning meaning
- Map columns to `X`, `Y`, and uncertainties
- Attach metadata and column names
- Copy, select, sort, and transform safely
- Handle uncertainties
- Fit one `dataArray`
- Validate and save a fit
- Use `dataList` for several datasets
- Avoid common mistakes

## Understand the `dataArray` layout

Treat a `dataArray` as a NumPy `ndarray` plus column roles and experimental metadata.
JScatter stores data columns on axis 0 and measurement points on axis 1:

```text
data.shape == (number_of_columns, number_of_points)

data[0]  -> all values in physical column 0
data[1]  -> all values in physical column 1
data[:, 5] -> every column for measurement point 5
```

This is the transpose of the common `(number_of_points, number_of_columns)` table
layout. Get this orientation right before interpreting or fitting anything.

JScatter assigns semantic aliases to selected rows. The complete alias order is:

```text
X, Y, eY, eX, Z, eZ, W, eW
```

`Y` is the dependent value used by fitting. `X`, optionally `Z` and `W`, are model
coordinates. Only `eY` weights ordinary fits; `eX`, `eZ`, and `eW` are retained but
are not propagated by the fitter.

Inspect the active mapping with:

```python
print(data.columnIndex)
# {'X': 0, 'Y': 1, 'eY': 2, 'eX': None,
#  'Z': None, 'eZ': None, 'W': None, 'eW': None}
```

The aliases are writable views of the mapped rows:

```python
data.X       # same physical values as data[0] for the mapping above
data.Y *= 2  # modifies the mapped row in data
data["Y"]   # explicit alias-based row access
```

Use `data.array` to obtain a plain NumPy view without JScatter attributes. Use
`data.array.copy()` when an independent NumPy array is required.

## Create one dataset from NumPy

Create a `(columns, points)` array explicitly. Prefer `np.vstack` when the input
variables already represent columns:

```python
import jscatter as js
import numpy as np

q = np.linspace(0.02, 0.5, 100)       # nm^-1
intensity = np.exp(-20.0 * q**2)      # cm^-1
sigma = np.full(q.size, 0.02)         # one-sigma uncertainty in cm^-1

data = js.dA(
    np.vstack((q, intensity, sigma)),
    XYeYeX=[0, 1, 2],
)
```

Transpose a conventional row-major table before constructing the `dataArray`:

```python
table = np.column_stack((q, intensity, sigma))  # shape (100, 3)
data = js.dA(table.T, XYeYeX=[0, 1, 2])         # shape (3, 100)
```

The default mapping is also `X=0`, `Y=1`, and `eY=2`. Still pass `XYeYeX` when
constructing non-obvious or multidimensional data so the meaning remains visible.

Be aware of identity and sharing:

- `js.dA(existing_data_array)` returns the same `dataArray`, not a copy.
- `js.dA(numpy_array)` may share the NumPy storage.
- Use `data.copy()` for an independent `dataArray` that preserves metadata and column
  mappings.

## Read one dataset from a file

Read one data block with `js.dA`; `index` is zero-based when a file contains several
blocks:

```python
data = js.dA("/inputs/measurement.dat")
third_block = js.dA("/inputs/multi-block.dat", index=2)
```

The generic ASCII reader normally interprets:

- a line beginning with at least two numbers as data;
- a name followed by one or more numbers as an attribute;
- other retained text as a comment;
- the source filename as `data.name`.

Read first and inspect before adding parser options. For a known file layout, select
source columns and immediately state their new mapping:

```python
data = js.dA(
    "/inputs/measurement.dat",
    usecols=[0, 2, 3],
    XYeYeX=[0, 1, 2],
)
```

Use `replace`, `skiplines`, `takeline`, `ignore`, `delimiter`, `block`, or
`lines2parameter` only after reading `help(js.dA)` and documenting why the file needs
that rule. `usecols` changes the in-memory row positions, so map roles after selection,
not from the original file indices.

Use `js.dL(path_or_glob)` instead when every block or multiple files must be retained.

When the task needs data the user did not supply, JScatter ships its own datasets
inside the installed package:

```python
data = js.dL(js.examples.datapath + "/exampledata.dat")
```

See `/skills/jscatter/references/example-data.md` for each bundled file with its verified
shape, columns, and loader — several need a non-obvious reader. Always prefer the
user's own data, and state when a result came from example data instead.

## Inspect before assigning meaning

Run this inspection before a fit or unit conversion:

```python
print("type", type(data))
print("shape", data.shape, "dtype", data.dtype)
print("column roles", data.columnIndex)
print("metadata names", data.attr)
data.showattr()
print("comments", data.comment[:5])
print("first points\n", data[:, :5])
```

Then verify explicitly:

1. Confirm `data.ndim == 2` and that axis 0 is truly the column axis.
2. Identify every mapped row from the file header, instrument format, or supplied
   experiment description.
3. Confirm coordinate, intensity, and uncertainty units.
4. Check that `X` and `Y` are finite in the intended range.
5. Check that `eY` is a measured or defensible one-sigma uncertainty, not merely the
   third numeric column.
6. Inspect metadata values used by a model, such as temperature, wavevector, contrast,
   wavelength, or sample concentration.

Do not infer column roles from magnitude alone.

## Map columns to `X`, `Y`, and uncertainties

Change roles without moving data:

```python
# File rows are: sigma_I, q, monitor, intensity
data.setColumnIndex(ix=1, iy=3, iey=0, iex=None)
print(data.columnIndex)
```

Disable an incorrect or absent uncertainty mapping explicitly:

```python
data.setColumnIndex(iey=None)
assert data.columnIndex["eY"] is None
```

Map two-dimensional coordinates when fitting detector-derived points:

```python
# Rows are qx, qz, intensity, sigma_I
image_points = js.dA(
    np.vstack((qx, qz, intensity, sigma_I)),
    XYeYeX=[0, 2, 3, None, 1],  # X, Y, eY, eX, Z
)
```

Never rely on a negative mapped index after adding rows: negative NumPy indices count
from the end and can silently refer to a different row.

## Attach metadata and column names

Store scalar or array metadata directly on the dataset with descriptive, model-friendly
names:

```python
data.temperature = 298.15
data.wavevector = 0.12
data.q_unit = "nm^-1"
data.intensity_unit = "cm^-1"
print(data.attr)
```

Avoid names already used by NumPy arrays, such as `T`, `mean`, `min`, or `max`, and
avoid punctuation in attribute names. Match a model parameter name to an attribute when
JScatter should pass that metadata automatically during fitting.

Numeric attributes round-trip automatically through JScatter's ASCII format. Arbitrary
string attributes are written as text but normally reread as comments because the parser
classifies `name string` differently from `name number`. Recover a required string with
`getfromcomment(...)` after reloading, or keep a documented unit convention in the
reproducible script. Dictionaries are not persisted by `data.save(...)`.

Optionally name physical rows independently of their fitting roles:

```python
data.columnname = "q; intensity; sigma_I"
print(data["intensity"])
print(data._intensity)
```

Treat `columnname` access as a readable label interface. Use `.X`, `.Y`, `.eY`, or
integer row indexing for deliberate mutation.

Extract retained nonnumeric metadata from comments only with a visible conversion rule:

```python
# Comment line: "Frequency MHz 3.141"
scale = {"GHz": 1e9, "MHz": 1e6, "kHz": 1e3, "Hz": 1.0}
data.getfromcomment(
    "Frequency",
    convert=lambda text: float(text.split()[1]) * scale[text.split()[0]],
)
```

## Copy, select, sort, and transform safely

Preserve the measured dataset before cleaning or transforming it:

```python
original = data.copy()
```

Select measurement points along axis 1 while retaining every row:

```python
mask = np.isfinite(data.X) & np.isfinite(data.Y)
if data.columnIndex["eY"] is not None:
    mask &= np.isfinite(data.eY) & (data.eY > 0)

prepared = data[:, mask].copy()
prepared = prepared.where(lambda a: (a.X >= qmin) & (a.X <= qmax))
prepared.isort(col="X")  # sorts in place
```

Combine array conditions with `&`, `|`, or `~`, and parenthesize each comparison. Do
not use Python `and` or `or` with NumPy arrays.

Transform values through mapped aliases so the intent is visible:

```python
kratky = prepared.copy()
kratky.Y = kratky.Y * kratky.X**2
kratky.eY = kratky.eY * kratky.X**2
```

Propagate uncertainties whenever a transformation changes the dependent values. Update
coordinate uncertainties as well when converting coordinate units.

## Handle uncertainties

Let JScatter use `eY` automatically only when it contains finite, positive one-sigma
uncertainties. A missing `eY` produces an unweighted fit. Zero values can also cause
equal weighting instead of the intended statistical weighting, so remove or justify
those points before fitting.

If a defensible uncertainty has been derived separately, append it as a new row and map
that row:

```python
sigma = propagated_standard_uncertainty
weighted = data.addColumn(values=sigma)  # returns a new dataArray
weighted.setColumnIndex(iey=weighted.shape[0] - 1)
```

Do not invent error bars merely to obtain parameter uncertainties. State explicitly
when a fit is unweighted or when an approximate statistical model, such as Poisson
counting uncertainty, is used.

## Fit one `dataArray`

Write a vectorized model with physical names. Do not name model arguments `X`, `Y`,
`eY`, `eX`, `Z`, `eZ`, `W`, or `eW`; those names are reserved by JScatter. They are,
however, the normal *targets* of `mapNames`: map a physical coordinate name such as `q`
or `t` onto its role, as in `mapNames={"q": "X"}`:

```python
def decay(t, amplitude, rate, background):
    return amplitude * np.exp(-rate * t) + background

prepared.setLimit(amplitude=(0.0, 10.0), rate=(0.0, 100.0))
prepared.fit(
    model=decay,
    freepar={"amplitude": 1.0, "rate": 0.2},
    fixpar={"background": 0.0},
    mapNames={"t": "X"},
    condition=lambda a: (a.X >= tmin) & (a.X <= tmax),
    method="trf",
)
```

Apply these rules:

- Return a one-dimensional modeled-Y array or a `dataArray` whose `.Y` has the same
  point count as the selected data.
- Do arithmetic on `.Y`, never on the whole `dataArray`. A `dataArray` holds every column
  (`X`, `Y`, `eY`, ...) as rows of one array, so `A * js.ff.sphere(q, R) + bgr` also
  scales and shifts `q` and the errors. Write `result.Y = A * result.Y + bgr`, or return
  `A * js.ff.sphere(q=q, radius=R).Y + bgr`.
- Use NumPy operations so the model accepts arrays.
- Pass `condition` as a callable such as `lambda a: (a.X > 0.05) & (a.X < 2)`, not a string.
- Put fitted starting values in `freepar` and fixed values in `fixpar`.
- Let same-named dataset metadata satisfy other model arguments, or map a differently
  named attribute with `mapNames`.
- Evaluate the model once at the initial values before fitting and require finite output.
- Use `lm` first for an unconstrained local least-squares fit. Use `trf` or `dogbox` for
  bounded least squares. Set limits with `setLimit` before fitting and inspect
  `prepared.hasLimit` (a property, no parentheses).
- `setLimit(par=(low, up, lowhard, uphard))`: two values set a lower *and* an upper soft
  limit, so `setLimit(bgr=[0, 1])` pulls `bgr` below 1. Use `(low, None)` for a lower
  bound only, and keep start values inside the limits.
- Treat `Nelder-Mead` and `differential_evolution` as optimization tools that do not
  supply the usual covariance-derived errors; refine locally when justified.

### `dataArray.fit` versus `dataList.fit`

`dataArray.fit` is a thin wrapper. It puts the single dataset into a one-element
`dataList`, calls `dataList.fit`, and copies `lastfit[0]` back. Its docstring says "only
first parameter is used if more than one given":

- each `freepar` value must be a number or a list; a list keeps only its first element,
  and a NumPy array is rejected;
- a `fixpar` list whose first element is a number is also cut to its first element;
- `method`, `weight`, and other options pass through `**kw` to `dataList.fit`.

The list, `[start]`, and `[start, "attribute_name"]` forms in the `dataList.fit`
docstring therefore only matter for a `dataList`. When explaining `dataArray.fit`, say
this explicitly instead of repeating the `dataList.fit` docstring as its contract.

## Validate and save a fit

After successful completion, JScatter stores the modeled result and diagnostics in
`prepared.lastfit` and copies fitted scalar parameters onto `prepared`:

```python
fit = prepared.lastfit
print("amplitude", fit.amplitude, "+/-", fit.amplitude_err)
print("rate", fit.rate, "+/-", fit.rate_err)
print("reduced chi2", fit.chi2)
print("degrees of freedom", fit.dof)
print("correlation matrix\n", fit.parcorr)
```

`fit.chi2` is already the reduced chi-square (the weighted squared residuals divided by
`dof`); do not divide it by `dof` again, and there is no `reduced_chi2` attribute.
`prepared.getFreepar` is a property that returns the best values in `freepar` form:
write `prepared.getFreepar`, not `prepared.getFreepar()`.

JScatter 1.9.0.1 does not store a generic optimizer `success` flag on `lastfit`.
Treat a fit exception, a reported failure, a `-1` result, or a missing `lastfit` as a
failed fit. Completion without those signals is necessary but not sufficient for a
scientifically valid result.

Do not accept parameters merely because the optimizer terminated. Check:

- that `lastfit` exists and all modeled values are finite;
- structured residuals with `showlastErrPlot()` or an explicit residual plot;
- finite parameter errors for solvers that provide them;
- large off-diagonal parameter correlations and weak identifiability;
- parameters on limits and sensitivity to starting values or fit range;
- reduced chi-square in the context of the uncertainty model, not as a score alone;
- physical units, contrast assumptions, and omitted model contributions.

Save measured/prepared data and fit results separately because saving `data` does not
automatically save `data.lastfit`:

```python
prepared.save("/workspace/outputs/prepared-data.dat")
prepared.lastfit.save("/workspace/outputs/fit-result.dat")
```

JScatter's ASCII format preserves numeric attributes, supported array attributes, and the
`XYeYeX` column mapping. Verify every required field after a round trip:

```python
reloaded = js.dA("/workspace/outputs/prepared-data.dat")
assert reloaded.columnIndex == prepared.columnIndex
assert np.isclose(reloaded.temperature, prepared.temperature)

# Recover a string line such as "q_unit nm^-1" when needed.
reloaded.getfromcomment("q_unit")
assert reloaded.q_unit == "nm^-1"
```

Use `np.savetxt(path, prepared.array.T)` only when a plain row-major table is explicitly
required; it discards JScatter metadata and column-role information.

## Use `dataList` for several datasets

Use a `dataList` for repeated measurements, multiple contrasts, temperatures,
wavevectors, or other datasets with different point counts:

```python
series = js.dL()
series.append(data_1)
series.append(data_2)

for dataset in series:
    print(dataset.shape, dataset.columnIndex, dataset.temperature)
```

For `dataList.fit` (these forms do not apply to a single `dataArray.fit`; see above):

- use a scalar parameter value for one value shared by all datasets;
- use `[start]` for an independent value per dataset;
- use `[start, "attribute_name"]` for one value per sorted unique metadata value;
- verify linked ordering with `np.unique(getattr(series, "attribute_name"))`;
- use dataset attributes as fixed model inputs and map coordinate roles explicitly.

Fit a single element with `series[index].fit(...)`. Save the simultaneous result with
`series.lastfit.save(...)`.

## Avoid common mistakes

- Do not pass a conventional `(points, columns)` table without transposing it.
- Do not assume row 2 is `eY`; the default mapping makes it so even when the third row
  is monitor counts, a second coordinate, or another signal.
- Do not treat `data.attr` as a dictionary; it is the list of stored attribute names.
- Do not assume `data.array` or `js.dA(existing_data_array)` creates a copy.
- Do not select points with `data[mask]`; retain the row axis with `data[:, mask]`.
- Do not overwrite measured data while exploring a correction; start from `data.copy()`.
- Do not use `.eX`, `.eZ`, or `.eW` as if the fitter propagated coordinate errors.
- Do not use protected aliases as model parameter names; map physical names to them.
- Do not save only the original dataset and expect `.lastfit` to be embedded.
- Do not apply arithmetic to a whole `dataArray` when you mean its intensity; use `.Y`.
- Do not describe `dataList.fit` list or link forms as the contract of `dataArray.fit`.
- Do not call properties such as `getFreepar` or `hasLimit` with parentheses.
