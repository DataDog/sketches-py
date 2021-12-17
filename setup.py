import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ddsketch",
    author="Jee Rim, Charles-Philippe Masson, Homin Lee",
    author_email="jee.rim@datadoghq.com, charles.masson@datadoghq.com, homin@datadoghq.com",
    description="Distributed quantile sketches",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/datadog/sketches-py",
    packages=setuptools.find_packages(),
    package_data={"ddsketch": ["py.typed"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    keywords=["ddsketch", "quantile", "sketch"],
    install_requires=[
        "protobuf>=3.14.0",
        "six",
        "typing; python_version<'3.5'",
    ],
    python_requires=">=2.7",
    download_url="https://github.com/DataDog/sketches-py/archive/v1.0.tar.gz",
    setup_requires=["setuptools_scm"],
    use_scm_version={"write_to": "ddsketch/__version.py"},
)
