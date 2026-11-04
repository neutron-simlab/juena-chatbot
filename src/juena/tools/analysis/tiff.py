"""Detector TIFF processor for the analyze_tiff agent tool.

This file is self-contained so delivered reproduction scripts need no JüNA install.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle
import numpy as np
import tifffile
import imagecodecs


MAX_PIXELS = 16_777_216
MAX_DECODED_BYTES = 128 * 1024**2
MAX_FILE_BYTES = 64 * 1024**2
MAX_INPUTS = 5
METADATA_TAGS = (
    "ImageDescription", "DateTime", "Software", "Make", "Model", "Orientation",
    "XResolution", "YResolution", "ResolutionUnit", "ExposureTime",
)


class TiffError(ValueError):
    """Unsupported input or operation; original files are never modified."""


def versions():
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "tifffile": tifffile.__version__, "imagecodecs": imagecodecs.__version__,
            "matplotlib": matplotlib.__version__}


def source_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_plane(path, page_index, remaining_bytes):
    path = Path(path)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise TiffError("TIFF input exceeds the 64 MiB file limit")
    with tifffile.TiffFile(path) as tif:
        count = len(tif.pages)
        if page_index < 0 or page_index >= count:
            raise TiffError(f"Page {page_index} is outside the zero-based range 0..{count - 1}")
        if tif.is_ome or tif.is_imagej or any(s.axes not in {"YX", "QYX", "IYX"} for s in tif.series):
            raise TiffError("Structured OME/ImageJ multidimensional layouts are not supported; supply a plain grayscale plane or stack")
        page = tif.pages[page_index]
        shape, dtype = page.shape, page.dtype
        if (len(shape) != 2 or page.axes != "YX" or page.samplesperpixel != 1
                or int(page.photometric) not in (0, 1)):
            raise TiffError(f"Unsupported layout {shape}, axes {page.axes}; expected one grayscale YX plane")
        if not ((dtype.kind in "ui" and dtype.itemsize in (1, 2, 4))
                or (dtype.kind == "f" and dtype.itemsize in (4, 8))):
            raise TiffError(f"Unsupported pixel type {dtype}")
        pixels = math.prod(shape)
        if pixels <= 0 or pixels > MAX_PIXELS:
            raise TiffError(f"Plane exceeds the {MAX_PIXELS:,}-pixel limit or is empty")
        decoded_bytes = pixels * dtype.itemsize
        if decoded_bytes > remaining_bytes:
            raise TiffError("Selected input planes exceed the 128 MiB decoded-array limit; select fewer or smaller planes")
        metadata = {}
        for name in METADATA_TAGS:
            tag = page.tags.get(name)
            if tag is not None:
                metadata[name] = str(tag.value)[:512] if tag.count <= 512 else "[omitted: tag exceeds 512 values]"
        info = {"path": str(path), "sha256": source_hash(path), "size_bytes": path.stat().st_size,
                "page": page_index, "page_count": count, "shape": list(shape), "axes": page.axes,
                "dtype": str(dtype), "byte_order": tif.byteorder, "bigtiff": tif.is_bigtiff,
                "compression": page.compression.name, "metadata": metadata}
        # Bound codec concurrency and decode only the requested page, never tif.asarray().
        data = page.asarray(maxworkers=1)
    if data.shape != shape or data.dtype != dtype:
        raise TiffError("Decoded plane does not match its declared shape or pixel type")
    return data, info


def finite_values(data):
    return data[np.isfinite(data)]


def statistics(data):
    values = finite_values(data)
    result = {"pixels": int(data.size), "finite": int(values.size),
              "nonfinite": int(data.size - values.size)}
    if not values.size:
        return {**result, **dict.fromkeys(("min", "max", "mean", "std", "sum"))}
    with np.errstate(over="ignore", invalid="ignore"):
        measures = {"min": float(values.min()), "max": float(values.max()),
                    "mean": float(values.mean(dtype=np.float64)),
                    "std": float(values.std(dtype=np.float64)),
                    "sum": float(values.sum(dtype=np.float64))}
    result.update({k: v if math.isfinite(v) else None for k, v in measures.items()})
    if any(v is None for v in result.values()):
        result["warning"] = "Some reductions overflowed float64 and are reported as null"
    return result


def display_limits(data, scale, bounds=None):
    values = finite_values(data)
    excluded = 0
    if scale == "log":
        excluded = int(np.count_nonzero(values <= 0))
        values = values[values > 0]
    if not values.size:
        return {"scale": scale, "low": None, "high": None, "excluded_nonpositive": excluded}
    low, high = (float(v) for v in (np.percentile(values, [1, 99]) if bounds is None else bounds))
    result = {"scale": scale, "low": low, "high": high,
              "clipped_below": int(np.count_nonzero(values < low)),
              "clipped_above": int(np.count_nonzero(values > high)),
              "excluded_nonpositive": excluded}
    if bounds is None:
        result["percentiles"] = [1, 99]
    return result


def draw_plane(ax, data, title, limits):
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    if limits["low"] is None:
        ax.text(.5, .5, "No displayable finite pixels", transform=ax.transAxes, ha="center")
        return
    low, high = limits["low"], limits["high"]
    if low == high:
        if limits["scale"] == "log":
            low, high = low / 2, high * 2
        else:
            delta = max(abs(low) * .01, .5)
            low, high = low - delta, high + delta
    norm = LogNorm(low, high) if limits["scale"] == "log" else Normalize(low, high)
    stride = max(1, math.ceil(max(data.shape) / 1024))
    thumbnail = data[::stride, ::stride]
    mask = ~np.isfinite(thumbnail)
    if limits["scale"] == "log":
        mask |= thumbnail <= 0
    im = ax.imshow(np.ma.array(thumbnail, mask=mask), origin="upper", norm=norm, cmap="viridis",
                   extent=(-.5, data.shape[1] - .5, data.shape[0] - .5, -.5), interpolation="nearest")
    ax.figure.colorbar(im, ax=ax, label="raw pixel value", shrink=.75)


def run(operation, inputs, *, pages=None, roi=None, scale="linear", output_dir=None,
        expected_hashes=None, expected_versions=None):
    if operation not in {"inspect", "histogram", "roi", "compare"} or scale not in {"linear", "log"}:
        raise TiffError("Unsupported operation or display scale")
    if not 1 <= len(inputs) <= MAX_INPUTS:
        raise TiffError("Supply between one and five TIFF inputs")
    if len(inputs) != (2 if operation == "compare" else 1) and operation != "inspect":
        raise TiffError("Comparison needs two inputs; histogram and ROI need one")
    if (operation == "roi") != (roi is not None):
        raise TiffError("Supply --roi x0 y0 x1 y1 only for the roi operation")
    pages = list(pages or [0])
    if len(pages) == 1:
        pages *= len(inputs)
    if len(pages) != len(inputs):
        raise TiffError("Supply one page index or one per input")
    if expected_versions is not None and versions() != expected_versions:
        raise TiffError(f"Reproduction requires these dependency versions: {expected_versions}")
    arrays, sources = [], []
    remaining = MAX_DECODED_BYTES
    for path, page in zip(inputs, pages):
        data, info = load_plane(path, page, remaining)
        if expected_hashes is not None and info["sha256"] != expected_hashes[len(sources)]:
            raise TiffError(f"Source hash changed: {path}")
        remaining -= data.nbytes
        info["statistics"] = statistics(data)
        info["display"] = display_limits(data, scale)
        arrays.append(data)
        sources.append(info)
    report = {"operation": operation, "versions": versions(), "sources": sources,
              "coordinates": "zero-based x,y; top-left origin; exclusive upper ROI bounds",
              "notes": ["Statistics use original finite pixel values; nonfinite pixels are counted separately.",
                        "Display scaling and thumbnail sampling do not change analysis data.",
                        "Units, calibration, saturation thresholds, and count semantics are not inferred."]}
    if operation == "roi":
        if len(roi) != 4 or any(type(v) is not int for v in roi):
            raise TiffError("ROI must contain four integer pixel bounds")
        x0, y0, x1, y1 = roi
        height, width = arrays[0].shape
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise TiffError("ROI must be nonempty and within the selected plane")
        report["roi"] = {"bounds": list(roi), "statistics": statistics(arrays[0][y0:y1, x0:x1])}
    if operation == "compare":
        if arrays[0].shape != arrays[1].shape:
            raise TiffError("Comparison requires equal-sized planes; no registration is applied")
        with np.errstate(over="ignore", invalid="ignore"):
            difference = np.subtract(arrays[1], arrays[0], dtype=np.float64)
        report["difference"] = {"definition": "second input minus first input",
                                "statistics": statistics(difference)}
        limits = [s["display"] for s in sources if s["display"]["low"] is not None]
        if limits:
            common = {"scale": scale, "low": min(d["low"] for d in limits),
                      "high": max(d["high"] for d in limits)}
            report["shared_display"] = common
            for data, info in zip(arrays, sources):
                info["display"] = display_limits(data, scale, (common["low"], common["high"]))
    out = Path(output_dir) if output_dir else Path("/workspace/outputs") / f"tiff-{uuid4().hex[:10]}"
    out.mkdir(parents=True, exist_ok=False)
    columns = 3 if operation == "compare" else min(2, len(inputs))
    rows = math.ceil((3 if operation == "compare" else len(inputs)) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(min(16, 5 * columns), 4 * rows), squeeze=False)
    try:
        if operation == "histogram":
            values = finite_values(arrays[0])
            low, high = (float(values.min()), float(values.max())) if values.size else (0., 1.)
            # Use float64 edges even for float32 pixels. Expand narrow ranges
            # enough to give all 100 bins distinct edges without changing pixels.
            padding = max(abs(low), abs(high)) * np.finfo(np.float64).eps * 100
            if low == high:
                padding = max(padding, .5)
            if high - low < 2 * padding:
                low, high = low - padding, high + padding
            counts, edges = np.histogram(values, bins=np.linspace(low, high, 101))
            report["histogram"] = {"counts": counts.tolist(), "edges": edges.tolist(), "bins": 100}
            axes.flat[0].stairs(counts, edges)
            axes.flat[0].set(xlabel="raw pixel value", ylabel="finite pixels", title=f"{Path(inputs[0]).name}, page {pages[0]}")
        else:
            for ax, data, info in zip(axes.flat, arrays, sources):
                draw_plane(ax, data, f"{Path(info['path']).name}, page {info['page']}", info["display"])
            if operation == "roi":
                axes.flat[0].add_patch(Rectangle((x0 - .5, y0 - .5), x1 - x0, y1 - y0, fill=False, edgecolor="red", linewidth=1))
            if operation == "compare":
                diff_display = display_limits(difference, "linear")
                report["difference"]["display"] = diff_display
                draw_plane(axes.flat[2], difference, "second − first", diff_display)
        for ax in list(axes.flat)[3 if operation == "compare" else len(inputs):]:
            ax.set_visible(False)
        fig.suptitle("Display only; raw values retained. Linear/log scaling recorded in report.", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "preview.png", dpi=110)
    finally:
        plt.close(fig)
    report["outputs"] = {"preview": str(out / "preview.png"), "report": str(out / "report.json"),
                         "reproduction": str(out / "reproduce.py")}
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    config = {"operation": operation, "inputs": list(map(str, inputs)), "pages": pages,
              "roi": roi, "scale": scale, "expected_hashes": [s["sha256"] for s in sources],
              "expected_versions": versions()}
    source = Path(__file__).read_text().split('\nif __name__ == "__main__":')[0]
    reproduction = source + '\nif __name__ == "__main__":\n    config = json.loads(' + repr(json.dumps(config)) + ')\n    config["output_dir"] = str(Path.cwd() / ("tiff-reproduced-" + uuid4().hex[:10]))\n    print(json.dumps(run(**config), allow_nan=False))\n'
    (out / "reproduce.py").write_text(reproduction)
    return report


def execute_request(payload):
    """Worker entry point for a validated, structured agent-tool request."""
    try:
        report = run(**payload)
    except (tifffile.TiffFileError, OSError, ValueError, RuntimeError, NotImplementedError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(report, allow_nan=False))
