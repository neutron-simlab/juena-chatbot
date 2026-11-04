# Detector files in chat

Upload `.tif` or `.tiff` detector images through the chat composer. JüNA gives new
TIFF uploads an automatic first look unless you ask to store them only: a preview
of page 0, image dimensions, pixel type, page count, metadata, and basic numerical
statistics. The original stays in the conversation's Files panel for downloading
and later analysis. Inspection requires the execution sandbox to be enabled.

Text-based `.dat` files are also accepted for numerical data or instrument output.
They must contain UTF-8 text. The extension alone does not specify columns or units;
provide these in your message when they are not documented in the file.

| Limit | Value |
| --- | --- |
| TIFF file | 64 MiB |
| Text file, including `.dat` | 1 MiB |
| Attachments per message | 5 |
| Combined attachment bytes per message | 96 MiB |
| Pixels per decoded TIFF plane | 16,777,216 |
| Decoded input arrays per analysis | 128 MiB |

The first release supports ordinary grayscale single-page and multi-page TIFFs,
including small BigTIFF files, integer pixels up to 32 bits, and 32/64-bit floating
point pixels. Color, complex-valued, and structured multidimensional images such
as OME/ImageJ hyperstacks need conversion to ordinary detector planes first.

Useful requests include:

- “Show frame 3 with a logarithmic display.” The tool uses zero-based pages;
  JüNA should state the corresponding page index when translating your request.
- “Plot the intensity histogram for page 0.” Histograms use 100 bins and original
  finite pixel values.
- “Measure the region x=10 through 99, y=20 through 149.” This corresponds to
  bounds `(10, 20, 100, 150)`, with a top-left origin and exclusive upper bounds.
- “Compare page 0 and page 1.” The difference is the second plane minus the first,
  without automatic registration or normalization.

Previews use first-to-99th-percentile display limits by default. The report records
the limits and clipping counts. Logarithmic display excludes nonpositive pixels
from the preview only. Numerical statistics preserve zero and negative values;
nonfinite values are counted separately. Neither metadata nor pixel values alone
establish physical units, calibration, saturation, or neutron counting semantics.

Each analysis delivers a PNG, a JSON report, and a standalone reproduction script.
The report records source SHA-256 hashes, page/region selections, scaling, and exact
dependency versions. The script checks those hashes and versions before repeating
the analysis and creates a new results folder. It requires the original files at
the recorded paths; adjust paths if reproducing on another computer while retaining
the hash checks. Dependencies are installed at image build time, never by the agent
while processing your file.

Calibrated scattering reduction still requires detector geometry, masks,
normalization, and uncertainty assumptions. A TIFF preview is not an I(q) curve.
