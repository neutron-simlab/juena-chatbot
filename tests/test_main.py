"""Tests for startup bootstrap behavior in main.py."""

from __future__ import annotations

import main


def test_main_runs_bootstrap_before_starting_server(monkeypatch) -> None:
    calls: list[object] = []

    monkeypatch.delenv(main.BOOTSTRAP_DONE_ENV, raising=False)
    monkeypatch.setenv("RELOAD", "false")
    monkeypatch.setattr(main, "bootstrap_repositories", lambda: calls.append("bootstrap"))
    monkeypatch.setattr(main.uvicorn, "run", lambda *args, **kwargs: calls.append(("uvicorn", kwargs)))

    main.main()

    assert calls[0] == "bootstrap"
    assert main.os.environ[main.BOOTSTRAP_DONE_ENV] == "1"
    assert calls[1] == (
        "uvicorn",
        {
            "host": main.global_config.SERVER_HOST,
            "port": main.global_config.SERVER_PORT,
            "reload": False,
            "log_level": "info",
        },
    )


def test_main_skips_bootstrap_when_flag_is_present(monkeypatch) -> None:
    calls: list[object] = []

    monkeypatch.setenv(main.BOOTSTRAP_DONE_ENV, "1")
    monkeypatch.setenv("RELOAD", "true")
    monkeypatch.setattr(
        main,
        "bootstrap_repositories",
        lambda: (_ for _ in ()).throw(AssertionError("bootstrap should be skipped")),
    )
    monkeypatch.setattr(main.uvicorn, "run", lambda *args, **kwargs: calls.append(("uvicorn", kwargs)))

    main.main()

    assert calls == [
        (
            "uvicorn",
            {
                "host": main.global_config.SERVER_HOST,
                "port": main.global_config.SERVER_PORT,
                "reload": True,
                "log_level": "info",
            },
        )
    ]
