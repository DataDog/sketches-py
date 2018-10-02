# sketches-py

This repo contains python implementations of the distributed quantile sketch algorithms GKArray [1]  and DogSketch [2]. Both sketches are fully mergeable, meaning that multiple sketches from distributed systems can be combined in a central node.

## Installation

To install this package, clone the repo and run `python setup.py install`. This package depends only on `numpy`.

## GKArray

GKArray provides a sketch with a rank error guarantee of espilon (without merge) or 2\*epsilon (with merge). The default value of epsilon is 0.005.

### Usage
```
from gkarray.gkarray import GKArray

sketch = GKArray()
```
Add some values to the sketch. 
```
import numpy as np
values = np.random.normal(size=500)
for v in values:
  sketch.add(v)
```
Find the quantiles of `values` to within epsilon of rank.
```
quantiles = sketch.quantiles([0.5, 0.75, 0.9, 1])
```
Merge another `GKArray` into `sketch`.
```
another_sketch = GKArray()
other_values = np.random.normal(size=500)
for v in other_values:
  another_sketch.add(v)
sketch.merge(another_sketch)
```
Now the quantiles of `values` concatenated with `other_values` will be accurate to within 2\*epsilon of rank.
```
quantiles = sketch.quantile([0.5, 0.75, 0.9, 1])
```

## DogSketch

DogSketch has a relative error guarantee of alpha for any quantile q in [0, 1] that is not too small. Concretely, the q-quantile will be accurate up to a relative error of alpha as long as it belongs to one of the m buckets kept by the sketch. The default values of alpha and m are 0.01 and 2048, repectively. In addition, a value that is smaller than min_value in magnitude is indistinguishable from 0. The default min_value is 1.0e-9.

### Usage
```
from dogsketch.dogsketch import DogSketch

sketch = DogSketch()
```
Add values to the sketch
```
import numpy as np

values = np.random.normal(size=500)
for v in values:
  sketch.add(v)
```
Find the quantiles of `values` to within alpha relative error.
```
quantiles = [sketch.quantile(q) for q in [0.5, 0.75, 0.9, 1]]
```
Merge another `DogSketch` into `sketch`.
```
another_sketch = DogSketch()
other_values = np.random.normal(size=500)
for v in other_values:
  another_sketch.add(v)
sketch.merge(another_sketch)
```
The quantiles of `values` concatenated with `other_values` are still accurate to within alpha relative error.
```
quantiles = [sketch.quantile(q) for q in [0.5, 0.75, 0.9, 1]]
```

## References
[1] Michael B. Greenwald and Sanjeev Khanna. Space-efficient online computation of quantile summaries. In Proc. 2001 ACM
SIGMOD International Conference on Management of Data, SIGMOD ’01, pages 58–66. ACM, 2001.

[2] Charles-Phillip Masson, Jee Rim and Homin K. Lee. All the nines: a fully mergeable quantile sketch with relative-error guarantees for arbitrarily large quantiles. 2018.

