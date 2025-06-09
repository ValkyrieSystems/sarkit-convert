import os
import pathlib

import nox

os.environ.update({"PDM_IGNORE_SAVED_PYTHON": "1"})

nox.options.sessions = (
    "lint",
    "test",
)


@nox.session
def format(session):
    session.run_install("pdm", "sync", "--prod", "-G", "dev-lint", external=True)
    session.run("ruff", "check", "--fix")
    session.run("ruff", "format")


@nox.session
def lint(session):
    session.run_install(
        "pdm", "sync", "--prod", "-G", "dev-lint", "-G", "all", external=True
    )
    session.run("ruff", "check")
    session.run(
        "ruff",
        "format",
        "--diff",
    )
    session.run("mypy", pathlib.Path(__file__).parent / "sarkit_convert")


@nox.session
def test(session):
    for next_session in (
        "test_core",
        "test_iceye",
        "test_cosmo",
        "test_tsx",
        "test_sentinel",
    ):
        session.notify(next_session)


@nox.session
def test_core(session):
    session.run_install(
        "pdm",
        "sync",
        "--prod",
        "-G",
        "dev-test",
        external=True,
    )
    session.run("pytest", "tests/core")


@nox.session
def test_iceye(session):
    session.run_install(
        "pdm",
        "sync",
        "--prod",
        "-G",
        "dev-test",
        "-G",
        "iceye",
        external=True,
    )
    session.run("pytest", "tests/iceye")


@nox.session
def test_cosmo(session):
    session.run_install(
        "pdm",
        "sync",
        "--prod",
        "-G",
        "dev-test",
        "-G",
        "cosmo",
        external=True,
    )
    session.run("pytest", "tests/cosmo")


@nox.session
def test_tsx(session):
    session.run_install(
        "pdm",
        "sync",
        "--prod",
        "-G",
        "dev-test",
        "-G",
        "tsx",
        external=True,
    )
    session.run("pytest", "tests/tsx")


@nox.session
def test_sentinel(session):
    session.run_install(
        "pdm",
        "sync",
        "--prod",
        "-G",
        "dev-test",
        "-G",
        "sentinel",
        external=True,
    )
    session.run("pytest", "tests/sentinel")
