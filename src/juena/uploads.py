"""The upload policy shared by JüNA's composer and enforcing API."""

from juena_core.schema.upload_limits import UploadPolicy


UPLOAD_POLICY = UploadPolicy(
    binary_suffixes=frozenset({".tif", ".tiff"}),
    max_binary_bytes=64 * 1024**2,
    max_total_bytes=96 * 1024**2,
)
