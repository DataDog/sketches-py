# ddsketch

This repo contains the Python implementation of the distributed quantile sketch
algorithm DDSketch [1]. DDSketch has relative-error guarantees for any quantile
q in [0, 1]. That is if the true value of the qth-quantile is `x` then DDSketch
returns a value `y` such that `|x-y| / x < e` where `e` is the relative error
parameter. (The default here is set to 0.01.)  DDSketch is also fully mergeable,
meaning that multiple sketches from distributed systems can be combined in a
central node.

Our default implementation, `DDSketch`, is guaranteed [1] to not grow too large
in size for any data that can be described by a distribution whose tails are
sub-exponential.

We also provide implementations (`LogCollapsingLowestDenseDDSketch` and
`LogCollapsingHighestDenseDDSketch`) where the q-quantile will be accurate up to
the specified relative error for q that is not too small (or large). Concretely,
the q-quantile will be accurate up to the specified relative error as long as it
belongs to one of the `m` bins kept by the sketch.  If the data is time in
seconds, the default of `m = 2048` covers 80 microseconds to 1 year.

## Installation

To install this package, run `pip install ddsketch`, or clone the repo and run
`python setup.py install`. This package depends on `numpy` and `protobuf`. (The
protobuf dependency can be removed if it's not applicable.)

## Usage
```
from ddsketch import DDSketch

sketch = DDSketch()
```
Add values to the sketch
```
import numpy as np

values = np.random.normal(size=500)
for v in values:
  sketch.add(v)
```
Find the quantiles of `values` to within the relative error.
```
quantiles = [sketch.get_quantile_value(q) for q in [0.5, 0.75, 0.9, 1]]
```
Merge another `DDSketch` into `sketch`.
```
another_sketch = DDSketch()
other_values = np.random.normal(size=500)
for v in other_values:
  another_sketch.add(v)
sketch.merge(another_sketch)
```
The quantiles of `values` concatenated with `other_values` are still accurate to within the relative error.

## Development

To work on ddsketch a Python interpreter must be installed. It is recommended to use the provided development
container (requires [docker](https://www.docker.com/)) which includes all the required Python interpreters.

    docker-compose run dev

Or, if developing outside of docker then it is recommended to use a virtual environment:

    pip install virtualenv
    virtualenv --python=3 .venv
    source .venv/bin/activate


### Testing

To run the tests install `riot`:

    pip install riot

Replace the Python version with the interpreter(s) available.

    # Run tests with Python 3.9
    riot run -p3.9 test

### Release notes

New features, bug fixes, deprecations and other breaking changes must have
release notes included.

To generate a release note for the change:

    riot run reno new <short-description-of-change-no-spaces>

Edit the generated file to include notes on the changes made in the commit/PR
and add commit it.


### Formatting

Format code with

    riot run fmt


### Type-checking

Type checking is done with [mypy](http://mypy-lang.org/):

    riot run mypy


### Type-checking

Lint the code with [flake8](https://flake8.pycqa.org/en/latest/):

    riot run flake8


## References
[1] Charles Masson and Jee E Rim and Homin K. Lee. DDSketch: A fast and fully-mergeable quantile sketch with relative-error guarantees. PVLDB, 12(12): 2195-2205, 2019. (The code referenced in the paper, including our implementation of the the Greenwald-Khanna (GK) algorithm, can be found at: https://github.com/DataDog/sketches-py/releases/tag/v0.1 )
