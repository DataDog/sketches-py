# Unless explicitly stated otherwise all files in this repository are licensed
# under the Apache License 2.0.
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2020 Datadog, Inc.

from collections import defaultdict, namedtuple
import unittest

import numpy as np

from datasets import *
from gkarray.gkarray import GKArray

test_eps = 0.05
test_quantiles = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1]
test_sizes = [3, 5, 10, 100, 1000]
datasets = [
    UniformForward,
    UniformBackward,
    UniformZoomIn,
    UniformZoomOut,
    UniformSqrt,
    Exponential,
    Lognormal,
    Normal,
    Laplace,
    Bimodal,
    Trimodal,
    Mixed,
]


def _evaluate_sketch_accuracy(sketch, data, eps):
    n = data.size
    for q in test_quantiles:
        sketch_rank = data.rank(sketch.quantile(q))
        data_rank = int(q * (n - 1) + 1)
        err = abs(sketch_rank - data_rank)
        np.testing.assert_equal(err - eps * n <= 0, True)
    np.testing.assert_equal(sketch.num_values, n)
    np.testing.assert_almost_equal(sketch.sum, data.sum)
    np.testing.assert_almost_equal(sketch.avg, data.avg)


class TestGKArray(unittest.TestCase):
    def test_distributions(self):
        for dataset in datasets:
            for n in test_sizes:
                data = dataset(n)
                sketch = GKArray(test_eps)
                for v in data.data:
                    sketch.add(v)
                _evaluate_sketch_accuracy(sketch, data, sketch.eps)

    def test_constant(self):
        for n in test_sizes:
            data = Constant(n)
            sketch = GKArray(test_eps)
            for v in data.data:
                sketch.add(v)
            for q in test_quantiles:
                np.testing.assert_equal(sketch.quantile(q), 42)

    def test_merge_equal(self):
        parameters = [(35, 1), (1, 3), (15, 2), (40, 0.5)]
        for n in test_sizes:
            d = EmptyDataset(0)
            s = GKArray(test_eps)
            for params in parameters:
                generator = Normal.from_params(params[0], params[1], n)
                sketch = GKArray(test_eps)
                for v in generator.data:
                    sketch.add(v)
                    d.add(v)
                s.merge(sketch)
            _evaluate_sketch_accuracy(s, d, 2 * s.eps)

    def test_merge_unequal(self):
        ntests = 20
        for i in range(ntests):
            for n in test_sizes:
                d = Lognormal(n)
                s1 = GKArray(test_eps)
                s2 = GKArray(test_eps)
                for v in d.data:
                    if np.random.random() > 0.7:
                        s1.add(v)
                    else:
                        s2.add(v)
                s1.merge(s2)
                _evaluate_sketch_accuracy(s1, d, 2 * s1.eps)

    def test_merge_mixed(self):
        ntests = 20
        datasets = [Normal, Exponential, Laplace, Bimodal]
        for i in range(ntests):
            d = EmptyDataset(0)
            s = GKArray(test_eps)
            for dataset in datasets:
                generator = dataset(np.random.randint(0, 500))
                sketch = GKArray(test_eps)
                for v in generator.data:
                    sketch.add(v)
                    d.add(v)
                s.merge(sketch)
            _evaluate_sketch_accuracy(s, d, 2 * s.eps)

    def test_consistent_merge(self):
        """Test that merge() calls do not modify the argument sketch."""
        s1 = GKArray(test_eps)
        s2 = GKArray(test_eps)
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
            [s2.quantile(q) for q in test_quantiles] + [s2.sum, s2.avg, s2.num_values],
            s2_summary,
        )

        s3 = GKArray(test_eps)
        s3.merge(s2)
        # merging to an empty sketch does not change s2
        np.testing.assert_almost_equal(
            [s2.quantile(q) for q in test_quantiles] + [s2.sum, s2.avg, s2.num_values],
            s2_summary,
        )
