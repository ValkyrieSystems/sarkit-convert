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
    session.run_install("pdm", "sync", external=True)
    session.run("ruff", "check", "--fix")
    session.run("ruff", "format")


@nox.session
def lint(session):
    session.run_install("pdm", "sync", "-G", "all", external=True)
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
        "test_core_dependencies",
        "test_iceye",
        "test_iceye_dependencies",
        "test_cosmo",
        "test_cosmo_dependencies",
        "test_tsx",
        "test_tsx_dependencies",
        "test_sentinel",
        "test_sentinel_dependencies",
    ):
        session.notify(next_session)


@nox.session
def test_core(session):
    session.run_install(
        "pdm",
        "sync",
        external=True,
    )
    session.run("pytest", "tests/core")


@nox.session
def test_core_dependencies(session):
    session.run_install(
        "pdm",
        "sync",
        "--prod",
        external=True,
    )
    session.run("python", "tests/core/test_dependencies.py")


@nox.session
def test_iceye(session):
    session.run_install(
        "pdm",
        "sync",
        "-G",
        "iceye",
        external=True,
    )
    session.run("pytest", "tests/iceye")


@nox.session
def test_iceye_dependencies(session):
    session.run_install(
        "pdm",
        "sync",
        "--prod",
        "-G",
        "iceye",
        external=True,
    )
    session.run("python", "tests/iceye/test_dependencies.py")


@nox.session
def test_cosmo(session):
    session.run_install(
        "pdm",
        "sync",
        "-G",
        "cosmo",
        external=True,
    )
    session.run("pytest", "tests/cosmo")


@nox.session
def test_cosmo_dependencies(session):
    session.run_install(
        "pdm",
        "sync",
        "--prod",
        "-G",
        "cosmo",
        external=True,
    )
    session.run("python", "tests/cosmo/test_dependencies.py")


@nox.session
def test_tsx(session):
    session.run_install(
        "pdm",
        "sync",
        "-G",
        "tsx",
        external=True,
    )
    session.run("pytest", "tests/tsx")


@nox.session
def test_tsx_dependencies(session):
    session.run_install(
        "pdm",
        "sync",
        "--prod",
        "-G",
        "tsx",
        external=True,
    )
    session.run("python", "tests/tsx/test_dependencies.py")


@nox.session
def test_sentinel(session):
    session.run_install(
        "pdm",
        "sync",
        "-G",
        "sentinel",
        external=True,
    )
    session.run("pytest", "tests/sentinel")


@nox.session
def test_sentinel_dependencies(session):
    session.run_install(
        "pdm",
        "sync",
        "--prod",
        "-G",
        "sentinel",
        external=True,
    )
    session.run("python", "tests/sentinel/test_dependencies.py")
