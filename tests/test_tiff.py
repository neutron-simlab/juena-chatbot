"""Numerical TIFF contracts and standalone reproducibility on real image fixtures."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import tifffile

from juena.tools.analysis import tiff


def image(tmp_path, values, name="sample.tif", **kwargs):
    path = tmp_path / name
    tifffile.imwrite(path, values, photometric="minisblack", **kwargs)
    return str(path)


@pytest.mark.parametrize("dtype", ["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64"])
def test_inspection_preserves_original_values_and_records_complete_provenance(tmp_path, dtype):
    values = np.array([[0, 1], [2, 3]], dtype=dtype)
    path = image(tmp_path, values)
    before = Path(path).read_bytes()
    result = tiff.run("inspect", [path], output_dir=tmp_path / "result")
    source = result["sources"][0]
    assert source["dtype"] == dtype
    assert source["sha256"] == hashlib.sha256(before).hexdigest()
    assert source["statistics"] == {"pixels": 4, "finite": 4, "nonfinite": 0,
                                    "min": 0, "max": 3, "mean": 1.5, "std": np.sqrt(1.25), "sum": 6}
    assert source["page"] == 0 and source["page_count"] == 1
    assert source["display"]["low"] == pytest.approx(.03)
    assert source["display"]["high"] == pytest.approx(2.97)
    assert set(result["versions"]) == {"python", "numpy", "tifffile", "imagecodecs", "matplotlib"}
    assert Path(result["outputs"]["preview"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert json.loads(Path(result["outputs"]["report"]).read_text()) == result
    assert Path(path).read_bytes() == before


@pytest.mark.parametrize("compression", [None, "lzw", "deflate", "packbits"])
@pytest.mark.parametrize("byteorder", ["<", ">"])
def test_compressed_bigtiff_and_byte_order(tmp_path, compression, byteorder):
    values = np.array([[0, 65535], [1, 40000]], dtype=np.uint16)
    path = image(tmp_path, values, compression=compression, byteorder=byteorder, bigtiff=True)
    data, info = tiff.load_plane(path, 0, tiff.MAX_DECODED_BYTES)
    np.testing.assert_array_equal(data, values)
    assert info["bigtiff"] and info["byte_order"] == byteorder


def test_page_selection_and_multi_input_first_look(tmp_path):
    path = image(tmp_path, np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.uint16))
    report = tiff.run("inspect", [path, path], pages=[0, 1], output_dir=tmp_path / "both")
    assert [s["statistics"]["sum"] for s in report["sources"]] == [6, 22]
    assert all(s["page_count"] == 2 for s in report["sources"])
    assert len(list((tmp_path / "both").iterdir())) == 3
    with pytest.raises(tiff.TiffError, match="outside"):
        tiff.run("inspect", [path], pages=[2])


def test_nonfinite_and_nonpositive_pixels_are_explicit_and_log_only_affects_display(tmp_path):
    values = np.array([[-1, 0, 1], [2, np.nan, np.inf]], dtype=np.float32)
    path = image(tmp_path, values)
    report = tiff.run("inspect", [path], scale="log", output_dir=tmp_path / "log")
    stats = report["sources"][0]["statistics"]
    assert stats["finite"] == 4 and stats["nonfinite"] == 2
    assert stats["min"] == -1 and stats["sum"] == 2 and stats["mean"] == .5
    assert report["sources"][0]["display"]["excluded_nonpositive"] == 2
    assert tiff.display_limits(np.array([[-2., 0]]), "log")["low"] is None


@pytest.mark.parametrize("value", [0., -2., np.nan])
def test_constant_and_all_nonfinite_images_produce_valid_reports_and_previews(tmp_path, value):
    path = image(tmp_path, np.full((2, 2), value))
    report = tiff.run("inspect", [path], output_dir=tmp_path / "constant")
    assert Path(report["outputs"]["preview"]).is_file()
    if np.isnan(value):
        assert report["sources"][0]["statistics"]["mean"] is None


def test_histogram_uses_original_values_and_accounts_for_every_finite_pixel(tmp_path):
    path = image(tmp_path, np.array([[-1., 0., 1.], [1., 2., np.nan]]))
    report = tiff.run("histogram", [path], output_dir=tmp_path / "hist")
    hist = report["histogram"]
    assert sum(hist["counts"]) == 5
    assert hist["edges"][0] == -1 and hist["edges"][-1] == 2
    assert len(hist["edges"]) == 101


def test_roi_has_top_left_origin_and_exclusive_upper_bounds(tmp_path):
    path = image(tmp_path, np.arange(12, dtype=np.int16).reshape(3, 4))
    report = tiff.run("roi", [path], roi=[1, 0, 3, 2], output_dir=tmp_path / "roi")
    assert report["roi"]["statistics"]["sum"] == 14
    assert report["roi"]["statistics"]["pixels"] == 4
    for bounds in ([-1, 0, 2, 1], [1, 0, 1, 1], [0, 0, 5, 1]):
        with pytest.raises(tiff.TiffError, match="ROI"):
            tiff.run("roi", [path], roi=bounds)


def test_unsigned_comparison_is_signed_and_uses_shared_scales(tmp_path):
    first = image(tmp_path, np.array([[65535, 1], [10, 20]], dtype=np.uint16), "a.tif")
    second = image(tmp_path, np.array([[0, 2], [10, 10]], dtype=np.uint16), "b.tif")
    report = tiff.run("compare", [first, second], output_dir=tmp_path / "diff")
    assert report["difference"]["statistics"]["min"] == -65535
    assert report["difference"]["statistics"]["sum"] == -65544
    assert report["shared_display"]["high"] >= max(s["display"]["high"] for s in report["sources"])
    different = image(tmp_path, np.zeros((3, 3), dtype=np.uint16), "different.tif")
    with pytest.raises(tiff.TiffError, match="equal-sized"):
        tiff.run("compare", [first, different])


@pytest.mark.parametrize("kind", ["color", "complex", "float16", "ome", "multidimensional"])
def test_unsupported_layouts_and_dtypes_fail_explicitly(tmp_path, kind):
    if kind == "color":
        path = tmp_path / "rgb.tif"
        tifffile.imwrite(path, np.zeros((3, 3, 3), dtype=np.uint8), photometric="rgb")
    else:
        values = np.zeros((2, 2), dtype={"complex": "complex64", "float16": "float16"}.get(kind, "uint16"))
        if kind == "multidimensional":
            values = np.zeros((2, 3, 4, 5), dtype=np.uint16)
        path = image(tmp_path, values, **({"ome": True} if kind == "ome" else {}))
    with pytest.raises(tiff.TiffError, match="[Uu]nsupported|multidimensional"):
        tiff.run("inspect", [str(path)])


def test_limits_reject_before_pixel_decoding(tmp_path, monkeypatch):
    path = image(tmp_path, np.zeros((2, 2), dtype=np.uint32))
    def forbidden(*args, **kwargs):
        pytest.fail("Pixel decoding must not run after a limit failure")
    monkeypatch.setattr(tifffile.TiffPage, "asarray", forbidden)
    with pytest.raises(tiff.TiffError, match="decoded-array"):
        tiff.load_plane(path, 0, 15)
    monkeypatch.setattr(tiff, "MAX_PIXELS", 3)
    with pytest.raises(tiff.TiffError, match="pixel limit"):
        tiff.load_plane(path, 0, 100)


def test_metadata_values_are_bounded(tmp_path):
    path = image(tmp_path, np.zeros((2, 2), dtype=np.uint8), description="x" * 10000)
    _, info = tiff.load_plane(path, 0, 100)
    assert len(info["metadata"]["ImageDescription"]) < 512


def test_reproduction_is_standalone_and_verifies_sources(tmp_path):
    path = image(tmp_path, np.array([[0., 2.], [4., 6.]]))
    result = tiff.run("inspect", [path], output_dir=tmp_path / "original")
    script = result["outputs"]["reproduction"]
    proc = subprocess.run([sys.executable, script], cwd=tmp_path, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    again = json.loads(proc.stdout)
    assert result["sources"] == again["sources"]
    assert result["versions"] == again["versions"]
    Path(path).write_bytes(Path(path).read_bytes() + b"changed")
    proc = subprocess.run([sys.executable, script], cwd=tmp_path, text=True, capture_output=True)
    assert proc.returncode != 0 and "Source hash changed" in proc.stderr


def test_malformed_tiff_returns_readable_worker_error_without_outputs(tmp_path, capsys):
    broken = tmp_path / "broken.tif"
    broken.write_bytes(b"not tiff")
    with pytest.raises(SystemExit) as error:
        tiff.execute_request({"operation": "inspect", "inputs": [str(broken)], "output_dir": str(tmp_path / "out")})
    assert error.value.code == 2
    assert json.loads(capsys.readouterr().err)["status"] == "error"
    assert not (tmp_path / "out").exists()
