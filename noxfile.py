import shutil

import nox

nox.options.default_venv_backend = "uv"

PYTHON_VERSIONS = ["3.11"]

# Prefer system uv over the potentially outdated one in .venv/bin/
UV = shutil.which("uv") or "uv"


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    session.run_install(
        UV,
        "pip",
        "install",
        ".[cpu,dev]",
        external=True,
    )
    session.run("pytest", "tests/", *session.posargs)
