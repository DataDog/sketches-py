from riot import Venv
from riot import latest


venv = Venv(
    pys=["3"],
    venvs=[
        Venv(
            name="test",
            command="pytest {cmdargs}",
            pys=["2.7", "3.6", "3.7", "3.8", "3.9", "3.10"],
            pkgs={
                "pytest": latest,
                "numpy": latest,
            },
        ),
        Venv(
            pkgs={
                "reno": latest,
            },
            venvs=[
                Venv(
                    name="reno",
                    command="reno {cmdargs}",
                )
            ],
        ),
        Venv(
            pkgs={
                "black": "==21.7b0",
                "isort": latest,
                "toml": latest,
            },
            venvs=[
                Venv(
                    name="black",
                    command="black {cmdargs}",
                ),
                Venv(
                    name="fmt",
                    command="isort . && black .",
                ),
                Venv(
                    name="check_fmt",
                    command="isort --check . && black --check .",
                ),
            ],
        ),
        Venv(
            name="mypy",
            command="mypy {cmdargs}",
            pkgs={
                "mypy": latest,
            },
        ),
    ],
)
