import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sketches",
    version="1.0.1",
    author="Jee Rim, Charles-Philippe Masson, Homin Lee",
    author_email="jee.rim@datadoghq.com, charles.masson@datadoghq.com, homin@datadoghq.com",
    description="Distributed quantile sketches",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/datadog/sketches-py",
    packages=["ddsketch", "gkarray"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    keywords=["ddsketch", "quantile", "sketch"],
    install_requires=[
        "numpy>=1.11.0",
        "protobuf>=3.14.0",
    ],
    python_requires=">=3.6",
    download_url="https://github.com/DataDog/sketches-py/archive/v1.0.tar.gz",
)
