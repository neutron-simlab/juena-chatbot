"""Cloning the repositories, and judging which ones are usable.

Everything about building or refreshing an index went with the Chroma cutover:
juena-rag owns ingestion, and this module now only clones and reports.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from juena.indexing import bootstrap


# --------------------------------------------------------------------------- #
# Surveying repositories -- one broken repository is not nine broken ones
# --------------------------------------------------------------------------- #


class _SurveyManager:
    """A repository manager whose clones are whatever the test says they are."""

    def __init__(self, cloned: set[str], repo_ids: list[str]) -> None:
        self._cloned = cloned
        self.repo_ids = repo_ids

    def has_local_root(self, repo_id: str) -> bool:
        return repo_id in self._cloned


class _SurveyIndex:
    """A search backend whose counts are whatever the test says they are."""

    def __init__(self, counts: dict[str, int | None], raises: Exception | None = None):
        self._counts = counts
        self._raises = raises

    def collection_count_existing(self, repo_id: str) -> int | None:
        if self._raises is not None:
            raise self._raises
        return self._counts.get(repo_id)


def test_a_repository_is_searchable_only_when_cloned_and_indexed() -> None:
    """Both halves: the store answers the search, the clone answers the read after it."""
    survey = bootstrap.survey_repositories(
        _SurveyManager({"a", "b"}, ["a", "b"]),
        _SurveyIndex({"a": 100, "b": 200}),
    )

    assert survey.searchable == ["a", "b"]
    assert survey.problems == []


def test_one_broken_repository_does_not_disable_the_others() -> None:
    """The whole point of step 4: `b` has no clone, so `a` and `c` still work."""
    survey = bootstrap.survey_repositories(
        _SurveyManager({"a", "c"}, ["a", "b", "c"]),
        _SurveyIndex({"a": 100, "b": 200, "c": 300}),
    )

    assert survey.searchable == ["a", "c"]
    assert survey.any_searchable
    assert survey.problems == ["repository cache missing for 'b'"]


def test_a_repository_in_the_store_but_not_on_disk_is_not_searchable() -> None:
    """It would answer the search and then fail the file read the model must do."""
    survey = bootstrap.survey_repositories(
        _SurveyManager(set(), ["a"]),
        _SurveyIndex({"a": 100}),
    )

    assert survey.searchable == []
    assert not survey.any_searchable


def test_a_repository_on_disk_but_empty_in_the_store_is_reported_as_empty() -> None:
    survey = bootstrap.survey_repositories(
        _SurveyManager({"a"}, ["a"]),
        _SurveyIndex({"a": 0}),
    )

    assert survey.searchable == []
    assert survey.problems == ["vector index empty for 'a'"]


def test_a_repository_absent_from_the_store_is_reported_as_missing() -> None:
    """None and zero are different answers, and this is the one that says None."""
    survey = bootstrap.survey_repositories(
        _SurveyManager({"a"}, ["a"]),
        _SurveyIndex({}),
    )

    assert survey.problems == ["vector index missing for 'a'"]


def test_an_unreachable_store_keeps_the_tools_and_says_so() -> None:
    """A network outage must not remove a capability for the life of the process.

    The compiled graph is cached, so a startup that decides "no repository tools"
    survives the outage that caused it. The tools stay; the failure is reported
    when a search is actually attempted.
    """
    survey = bootstrap.survey_repositories(
        _SurveyManager({"a", "b"}, ["a", "b"]),
        _SurveyIndex({}, raises=ConnectionError("cannot reach the juena-rag service")),
    )

    assert survey.searchable == ["a", "b"]
    assert survey.any_searchable
    assert "search backend unreachable" in survey.problems[0]


def test_no_repositories_configured_is_its_own_answer() -> None:
    survey = bootstrap.survey_repositories(_SurveyManager(set(), []), _SurveyIndex({}))

    assert survey.searchable == []
    assert survey.problems == ["no repositories configured"]


def _stub_manager(repo_ids: list[str], resolved: list[tuple[str, str | None]],
                  broken: set[str] = frozenset()):
    """A repository manager that records which revision each clone was asked for."""

    class StubRepoManager:
        def __init__(self, _configs: object) -> None:
            self.repo_ids = repo_ids

        def resolve_root(self, repo_id: str, revision: str | None = None) -> Path:
            if repo_id in broken:
                raise RuntimeError("git said no")
            resolved.append((repo_id, revision))
            return Path("/tmp") / repo_id

        def current_revision(self, repo_id: str) -> str:
            return f"{repo_id}-sha"

    return StubRepoManager


def test_cloning_refreshes_every_repository_and_builds_no_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The half this pod still needs once search is remote."""
    resolved: list[tuple[str, str | None]] = []
    monkeypatch.setattr(bootstrap, "load_repo_configs", lambda _p=None: [])
    monkeypatch.setattr(bootstrap, "RepoManager", _stub_manager(["a", "b"], resolved))
    monkeypatch.setattr(bootstrap, "indexed_revisions", dict)

    revisions = bootstrap.clone_repositories()

    assert [repo_id for repo_id, _ in resolved] == ["a", "b"]
    assert revisions == {"a": "a-sha", "b": "b-sha"}


def test_each_clone_is_checked_out_at_the_revision_juena_rag_indexed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """So a hit's line numbers are the lines of the file the model opens."""
    resolved: list[tuple[str, str | None]] = []
    monkeypatch.setattr(bootstrap, "load_repo_configs", lambda _p=None: [])
    monkeypatch.setattr(bootstrap, "RepoManager", _stub_manager(["a", "b"], resolved))
    monkeypatch.setattr(bootstrap, "indexed_revisions", lambda: {"a": "a-indexed"})

    bootstrap.clone_repositories()

    # `b` is not in the store, so there is no revision to match: it follows its branch.
    assert resolved == [("a", "a-indexed"), ("b", None)]


def test_clones_follow_their_branches_when_juena_rag_cannot_say(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved: list[tuple[str, str | None]] = []
    monkeypatch.setattr(bootstrap, "load_repo_configs", lambda _p=None: [])
    monkeypatch.setattr(bootstrap, "RepoManager", _stub_manager(["a"], resolved))

    def unreachable():
        raise ConnectionError("cannot reach the juena-rag service")

    monkeypatch.setattr(bootstrap, "build_search_index", lambda: SimpleNamespace(
        indexed_revisions=unreachable))

    assert bootstrap.clone_repositories() == {"a": "a-sha"}
    assert resolved == [("a", None)]


def test_a_repository_that_will_not_clone_does_not_stop_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved: list[tuple[str, str | None]] = []
    monkeypatch.setattr(bootstrap, "load_repo_configs", lambda _p=None: [])
    monkeypatch.setattr(
        bootstrap, "RepoManager", _stub_manager(["broken", "fine"], resolved, {"broken"})
    )
    monkeypatch.setattr(bootstrap, "indexed_revisions", dict)

    assert bootstrap.clone_repositories() == {"fine": "fine-sha"}
