---
name: jscatter
description: Analyze, model, fit, and plot neutron and X-ray scattering data with JScatter. Use for JScatter code or data involving dataArray/dataList, simultaneous and constrained fits, SANS/SAXS reduction and resolution smearing, form and structure factors, NSE/QENS or other dynamics, biomolecular scattering, DLS/CONTIN, headless plots, result validation, bundled example datasets for demonstrations, or debugging JScatter APIs and workflows.
---

# JScatter

Use JScatter to turn scattering data and physical assumptions into a reproducible,
validated analysis. Target the installed package rather than relying on remembered
signatures.

## Establish the evidence baseline

1. Inspect the staged inputs that the task depends on. Never alter `/inputs`. When
   the task needs data and none was supplied, JScatter ships its own example
   datasets under `js.examples.datapath` — see
   `/skills/jscatter/references/example-data.md`. Use the user's data whenever it
   exists, and state plainly whenever a result came from bundled example data
   instead.
2. If executing, import `jscatter as js` and record `js.__version__` before analysis.
3. Treat version `1.9.0.1` as this skill's documented baseline. If the installed
   version differs, disclose the mismatch and verify every used signature against
   `help(...)`, its docstring, or matching source before continuing.
4. Prefer exact version-matched evidence in `/repos/jscatter/src/jscatter/doc/source/`
   and `/repos/jscatter/src/jscatter/`. These are virtual paths for `read_file`,
   `grep`, `glob`, and indexed repository tools only. They do not exist inside the
   execution container: never pass `/repos` to `execute`, `cd`, Python imports,
   subprocesses, or script filenames. Use the condensed references for workflow
   guidance, not as a substitute for an exact callable signature.
5. Never invent an instrument setting, unit, column meaning, uncertainty, contrast,
   sample composition, or physical constraint. Request or state missing assumptions.

## Route to the minimum reference set

- Read `/skills/jscatter/references/data-and-fitting.md` for data ingestion,
  `dataArray`/`dataList`, model functions, fit parameters, solvers, uncertainties,
  result validation, and persistence.
- Read `/skills/jscatter/references/sas-and-static-models.md` for SANS/SAXS
  reduction, detector images, SLD/contrast, form factors, structure factors,
  polydispersity, lattices, and resolution smearing.
- Read `/skills/jscatter/references/dynamics-bio-and-dls.md` for NSE, QENS,
  time/frequency transforms, resolution convolution, biomolecules, deuteration,
  diffusion, normal modes, and DLS/CONTIN.
- Read `/skills/jscatter/references/execution-recipes.md` only when writing or
  running JScatter code, producing plots, or saving fit results.
- Read `/skills/jscatter/references/example-data.md` when the task needs data that
  the user did not supply — a demonstration, a reproduction of a documented
  workflow, or a check that a model behaves before applying it to real data. It
  lists each bundled file with its verified shape, columns, and loader, including
  the several files that need a non-obvious reader.

Read more than one reference only when the task genuinely crosses those boundaries.

## Apply the analysis workflow

1. Inspect shapes, attributes, column indices, finite values, units, and error columns
   before manipulating or fitting data. Preserve an untouched representation of the
   measured values.
2. Choose the smallest physical model justified by the experiment. Separate instrument
   corrections, form factor, structure factor, dynamics, resolution, scale, and
   background so each assumption remains visible.
3. Verify model parameter names and return shape. Map coordinates and metadata
   explicitly when their names do not match the model.
4. Use physically justified initial values, limits, shared parameters, and per-dataset
   parameters. Do not add constraints merely to force convergence.
5. Fit, confirm that JScatter reported no failure and produced `lastfit`, then check
   residual structure, fitted uncertainties, parameter correlations or identifiability,
   boundary hits, and unit consistency. Never report parameters from a failed fit as
   valid results.
6. Prefer fitting a resolution-smeared model to desmearing noisy SAS data. Desmear only
   when the user requests it and the edge extrapolation assumptions are defensible.
7. Save the model code, processed data, fit result including errors, and diagnostic plot
   needed to reproduce the conclusion.

## Execute within JueNA's sandbox

- Inspect `/repos` and `/skills` only through the specialist's read tools. `/inputs`
  is also mounted read-only inside `execute`; use it for staged inputs and use
  `/workspace` for persistent working files and `/workspace/.scratch` for disposable execution scripts. Never assume the virtual `/repos` or `/skills`
  routes are mounted in the execution container. Use `/workspace/outputs` for
  requested or reusable artifacts.
- Do not install packages or access the network. JScatter is already installed.
- Call `js.headless(True)` immediately after importing JScatter and before any fit or
  `GracePlot` construction. Fits can launch live xmgrace progress plotting even when
  the script does not explicitly create a plot. Use JScatter's native `GracePlot`
  by default and save an editable `.agr` plus a PNG when useful; its required
  `gracebat` executable is preinstalled. Use Matplotlib's non-interactive object API
  for plots that need more complex or portable customization.
- `write_file` only creates a new scratch path. When correcting a script that already
  exists, use `edit_file` or a new versioned filename; do not retry `write_file` with
  the same path.
- Use explicit output filenames and include units in axes and saved metadata.
- After a successful command reports collected artifacts, stop executing and return
  the specialist report. Do not re-read or encode collected output files.

## Report the result

Name the JScatter version, data selection, model components, fitted/shared parameters,
resolution treatment, solver, fit quality, uncertainties, and remaining physical
ambiguities. Cite exact repository paths or named JScatter documentation pages for
non-obvious API claims. List generated files through the specialist artifact contract.
