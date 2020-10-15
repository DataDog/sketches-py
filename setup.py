from setuptools import find_packages, setup

setup(
    name="sketches",
    version="0.1",
    description="Distributed quantile sketches",
    url="http://github.com/datadog/sketches-py",
    author="Jee Rim, Charles-Philippe Masson",
    author_email="jee.rim@datadoghq.com charles.masson@datadoghq.com",
    license="BSD-3-Clause",
    packages=["ddsketch", "gkarray"],
    install_requires=["numpy>=1.11.0"],
)
