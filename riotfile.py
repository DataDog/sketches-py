from riot import Venv
from riot import latest


venv = Venv(
    pys=["3"],
    venvs=[
        Venv(
            name="test",
            command="pytest {cmdargs}",
            pkgs={
                "pytest": latest,
                "numpy": latest,
            },
            venvs=[
                Venv(
                    pys=["2.7", "3.6"],
                    pkgs={
                        "protobuf": [
                            "==3.0.0",
                            "<3.19",
                            "!=4.21.0",
                        ],  # not latest due to https://github.com/protocolbuffers/protobuf/issues/10053
                    },
                ),
                Venv(
                    pys=["3.7", "3.8", "3.9"],
                    pkgs={
                        "protobuf": ["==3.0.0", "<3.19", latest],
                    },
                ),
                Venv(
                    pys=["3.10", "3.11"],
                    pkgs={
                        "protobuf": ["==3.8.0", "<3.19.0", latest],
                    },
                ),
            ],
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
            name="flake8",
            command="flake8 {cmdargs}",
            pkgs={
                "flake8": latest,
                "flake8-blind-except": latest,
                "flake8-builtins": latest,
                "flake8-docstrings": latest,
                "flake8-logging-format": latest,
                "flake8-rst-docstrings": latest,
                # needed for some features from flake8-rst-docstrings
                "pygments": latest,
            },
        ),
        Venv(
            pkgs={
                "black": latest,
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
            create=True,
            command="mypy --install-types --non-interactive {cmdargs}",
            pkgs={
                "mypy": latest,
                "types-protobuf": latest,
                "types-setuptools": latest,
                "types-six": latest,
            },
        ),
    ],
)
