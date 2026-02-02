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
    session.run_install("pdm", "sync", "-G", "dev-lint", external=True)
    session.run("ruff", "check", "--fix")
    session.run("ruff", "format")


@nox.session
def lint(session):
    session.run_install("pdm", "sync", "-G", "all", "-G", "dev-lint", external=True)
    session.run("ruff", "check")
    session.run(
        "ruff",
        "format",
        "--diff",
    )
    session.run("mypy", pathlib.Path(__file__).parent / "sarkit_convert")


@nox.session(
    requires=[
        "test_core",
        "test_core_dependencies",
        "test_extra",
        "test_extra_dependencies",
    ]
)
def test(session):
    """Run the required tests"""


@nox.session
def test_core(session):
    session.run_install(
        "pdm",
        "sync",
        "-G",
        "dev-test",
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


EXTRAS_TO_TEST = ["cosmo", "iceye", "sentinel", "terrasar"]


@nox.session
@nox.parametrize("name", EXTRAS_TO_TEST)
def test_extra(session, name):
    session.run_install(
        "pdm",
        "sync",
        "-G",
        name,
        "-G",
        "dev-test",
        external=True,
    )
    session.run("pytest", f"tests/{name}")


@nox.session
@nox.parametrize("name", EXTRAS_TO_TEST)
def test_extra_dependencies(session, name):
    session.run_install(
        "pdm",
        "sync",
        "--prod",
        "-G",
        name,
        external=True,
    )
    session.run("python", f"tests/{name}/test_dependencies.py")
