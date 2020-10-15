# Unless explicitly stated otherwise all files in this repository are licensed
# under the Apache License 2.0.
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2020 Datadog, Inc.

from collections import defaultdict, namedtuple

import numpy as np

from datasets import *
from ddsketch.ddsketch import DDSketch

test_quantiles = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1]
test_sizes = [3, 5, 10, 100, 1000]
datasets = [
    UniformForward,
    UniformBackward,
    UniformZoomIn,
    UniformZoomOut,
    UniformSqrt,
    Constant,
    Exponential,
    Lognormal,
    Normal,
    Laplace,
    Bimodal,
    Trimodal,
    Mixed,
]

test_alpha = 0.05
test_bin_limit = 1024
test_min_value = 1.0e-9


def test_distributions():
    for dataset in datasets:
        for n in test_sizes:
            data = dataset(n)
            sketch = DDSketch(test_alpha, test_bin_limit, test_min_value)
            for v in data.data:
                sketch.add(v)
            evaluate_sketch_accuracy(sketch, data, test_alpha)


def evaluate_sketch_accuracy(sketch, data, eps):
    n = data.size
    for q in test_quantiles:
        sketch_q = sketch.quantile(q)
        data_q = data.quantile(q)
        err = abs(sketch_q - data_q)
        np.testing.assert_equal(err - eps * abs(data_q) <= 1e-15, True)
    np.testing.assert_equal(sketch.num_values, n)
    np.testing.assert_almost_equal(sketch.sum, data.sum)
    np.testing.assert_almost_equal(sketch.avg, data.avg)


def test_merge_equal():
    parameters = [(35, 1), (1, 3), (15, 2), (40, 0.5)]
    for n in test_sizes:
        d = EmptyDataset(0)
        s = DDSketch(test_alpha, test_bin_limit, test_min_value)
        for params in parameters:
            generator = Normal.from_params(params[0], params[1], n)
            sketch = DDSketch(test_alpha, test_bin_limit, test_min_value)
            for v in generator.data:
                sketch.add(v)
                d.add(v)
            s.merge(sketch)
        evaluate_sketch_accuracy(s, d, test_alpha)


def test_merge_unequal():
    ntests = 20
    for i in range(ntests):
        for n in test_sizes:
            d = Lognormal(n)
            s1 = DDSketch(test_alpha, test_bin_limit, test_min_value)
            s2 = DDSketch(test_alpha, test_bin_limit, test_min_value)
            for v in d.data:
                if np.random.random() > 0.7:
                    s1.add(v)
                else:
                    s2.add(v)
            s1.merge(s2)
            evaluate_sketch_accuracy(s1, d, test_alpha)


def test_merge_mixed():
    ntests = 20
    datasets = [Normal, Exponential, Laplace, Bimodal]
    for i in range(ntests):
        d = EmptyDataset(0)
        s = DDSketch(test_alpha, test_bin_limit, test_min_value)
        for dataset in datasets:
            generator = dataset(np.random.randint(0, 500))
            sketch = DDSketch(test_alpha, test_bin_limit, test_min_value)
            for v in generator.data:
                sketch.add(v)
                d.add(v)
            s.merge(sketch)
        evaluate_sketch_accuracy(s, d, test_alpha)


def test_consistent_merge():
    """Test that merge() calls do not modify the argument sketch."""
    s1 = DDSketch(test_alpha, test_bin_limit, test_min_value)
    s2 = DDSketch(test_alpha, test_bin_limit, test_min_value)
    d = Normal(100)
    for v in d.data:
        s1.add(v)
    s1.merge(s2)
    # s2 is still empty
    np.testing.assert_equal(s2.num_values, 0)

    d = Normal(50)
    for v in d.data:
        s2.add(v)

    s2_summary = [s2.quantile(q) for q in test_quantiles] + [
        s2.sum,
        s2.avg,
        s2.num_values,
    ]
    s1.merge(s2)
    d = Normal(10)
    for v in d.data:
        s1.add(v)
    # changes to s1 does not affect s2 after merge
    s2_summary = [s2.quantile(q) for q in test_quantiles] + [
        s2.sum,
        s2.avg,
        s2.num_values,
    ]
    np.testing.assert_almost_equal(
        s2_summary,
        [s2.quantile(q) for q in test_quantiles] + [s2.sum, s2.avg, s2.num_values],
    )

    s3 = DDSketch(test_alpha, test_bin_limit, test_min_value)
    s3.merge(s2)
    # merging to an empty sketch does not change s2
    np.testing.assert_almost_equal(
        s2_summary,
        [s2.quantile(q) for q in test_quantiles] + [s2.sum, s2.avg, s2.num_values],
    )
