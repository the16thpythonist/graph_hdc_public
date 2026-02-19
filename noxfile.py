import nox

nox.options.default_venv_backend = "uv"

PYTHON_VERSIONS = ["3.11"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    session.install(".[cpu,dev]")
    session.run("pytest", "tests/", *session.posargs)
