from setuptools import find_packages, setup

setup(
    name="sketches",
    version="0.1",
    description="Distributed quantile sketches",
    url="http://github.com/datadog/sketches-py",
    author="Jee Rim, Charles-Philippe Masson, Homin Lee",
    author_email="jee.rim@datadoghq.com charles.masson@datadoghq.com homin@datadoghq.com",
    license="Apache License 2.0",
    packages=["ddsketch", "gkarray"],
    install_requires=["numpy>=1.11.0", "protobuf>=3.14.0"],
)
