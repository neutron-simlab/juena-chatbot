"""Backward-compatible re-export – canonical location is juena.indexing.bootstrap."""

from juena.indexing import bootstrap as _canonical  # noqa: E402

BOOTSTRAP_DONE_ENV = _canonical.BOOTSTRAP_DONE_ENV
bootstrap_repositories = _canonical.bootstrap_repositories
validate_bootstrap_ready = _canonical.validate_bootstrap_ready
main = _canonical.main

load_repo_configs = _canonical.load_repo_configs
RepoManager = _canonical.RepoManager
RepoVectorIndex = _canonical.RepoVectorIndex

if __name__ == "__main__":
    raise SystemExit(main())
