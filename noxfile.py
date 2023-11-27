import os

import nox

os.environ.update({"PDM_IGNORE_SAVED_PYTHON": "1"})


@nox.session(reuse_venv=True)
def tests(session):
    session.run("pdm", "install", "-dG", "test", external=True)
    session.run("pytest")


@nox.session(reuse_venv=True)
def lint(session):
    session.run("pdm", "install", "-dG", "qa", external=True)
    session.run("ruff", "met")


@nox.session(reuse_venv=True)
def type_check(session):
    session.run("pdm", "install", "-dG", "qa", external=True)
    session.run("pyright", "met")


# Uncomment if not using GitHub Actions to build
# @nox.session(python=False, tags=["docs", "pre-release"])
# def docs(session):
#     session.run("mkdocs", "gh-deploy")


@nox.session(reuse_venv=True)
def check_manifest(session):
    session.run("check-manifest", ".")
