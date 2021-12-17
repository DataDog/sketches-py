# Unless explicitly stated otherwise all files in this repository are licensed
# under the Apache License 2.0.
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2020 Datadog, Inc.

"""Tests for DDSketch"""

import abc
from collections import Counter
from unittest import TestCase

import numpy as np
import pytest
import six

import ddsketch
from ddsketch.ddsketch import DDSketch
from ddsketch.ddsketch import LogCollapsingHighestDenseDDSketch
from ddsketch.ddsketch import LogCollapsingLowestDenseDDSketch
from tests.datasets import Bimodal
from tests.datasets import Constant
from tests.datasets import EmptyDataset
from tests.datasets import Exponential
from tests.datasets import Integers
from tests.datasets import Laplace
from tests.datasets import Lognormal
from tests.datasets import Mixed
from tests.datasets import NegativeUniformBackward
from tests.datasets import NegativeUniformForward
from tests.datasets import Normal
from tests.datasets import NumberLineBackward
from tests.datasets import NumberLineForward
from tests.datasets import Trimodal
from tests.datasets import UniformBackward
from tests.datasets import UniformForward
from tests.datasets import UniformSqrt
from tests.datasets import UniformZoomIn
from tests.datasets import UniformZoomOut


TEST_QUANTILES = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1]
TEST_SIZES = [3, 5, 10, 100, 1000]
DATASETS = [
    UniformForward,
    UniformBackward,
    UniformZoomIn,
    UniformZoomOut,
    UniformSqrt,
    Constant,
    NegativeUniformBackward,
    NegativeUniformForward,
    NumberLineBackward,
    NumberLineForward,
    Exponential,
    Lognormal,
    Normal,
    Laplace,
    Bimodal,
    Trimodal,
    Mixed,
    Integers,
]

TEST_REL_ACC = 0.05
TEST_BIN_LIMIT = 1024


class BaseTestDDSketches(six.with_metaclass(abc.ABCMeta)):
    """AbstractBaseClass for testing DDSketch implementations"""

    @staticmethod
    @abc.abstractmethod
    def _new_dd_sketch():
        """Create a new DDSketch of the appropriate type"""

    def _evaluate_sketch_accuracy(self, sketch, data, eps, summary_stats=True):
        size = data.size
        for quantile in TEST_QUANTILES:
            sketch_q = sketch.get_quantile_value(quantile)
            data_q = data.quantile(quantile)
            err = abs(sketch_q - data_q)
            assert err - eps * abs(data_q) <= 1e-15
        assert sketch.num_values == size
        if summary_stats:
            assert sketch.sum == pytest.approx(data.sum)
            assert sketch.avg == pytest.approx(data.avg)

    def test_distributions(self):
        """Test DDSketch on values from various distributions"""
        for dataset in DATASETS:
            for size in TEST_SIZES:
                data = dataset(size)
                sketch = self._new_dd_sketch()
                for value in data.data:
                    sketch.add(value)
                self._evaluate_sketch_accuracy(sketch, data, TEST_REL_ACC)

    def test_add_multiple(self):
        """Test DDSketch on adding integer weighted values"""
        data = Integers(1000)
        sketch = self._new_dd_sketch()
        for value, count in Counter(data.data).items():
            sketch.add(value, count)
        self._evaluate_sketch_accuracy(sketch, data, TEST_REL_ACC)

    def test_add_decimal(self):
        """Test DDSketch on adding decimal weighted values"""
        sketch = self._new_dd_sketch()
        for value in range(100):
            sketch.add(value, 1.1)
        sketch.add(100, 110.0)

        data_median = 99
        sketch_median = sketch.get_quantile_value(0.5)
        err = abs(sketch_median - data_median)
        assert err - TEST_REL_ACC * abs(data_median) <= 1e-15
        assert sketch.num_values == pytest.approx(110 * 2)
        assert sketch.sum == pytest.approx(5445 + 11000)
        assert sketch.avg == pytest.approx(74.75)

    def test_merge_equal(self):
        """Test merging equal-sized DDSketches"""
        parameters = [(35, 1), (1, 3), (15, 2), (40, 0.5)]
        for size in TEST_SIZES:
            dataset = EmptyDataset(0)
            target_sketch = self._new_dd_sketch()
            for params in parameters:
                generator = Normal.from_params(params[0], params[1], size)
                sketch = self._new_dd_sketch()
                for value in generator.data:
                    sketch.add(value)
                    dataset.add(value)
                target_sketch.merge(sketch)
                self._evaluate_sketch_accuracy(target_sketch, dataset, TEST_REL_ACC)

            self._evaluate_sketch_accuracy(target_sketch, dataset, TEST_REL_ACC)

    def test_merge_unequal(self):
        """Test merging variable-sized DDSketches"""
        ntests = 20
        for _ in range(ntests):
            for size in TEST_SIZES:
                dataset = Lognormal(size)
                sketch1 = self._new_dd_sketch()
                sketch2 = self._new_dd_sketch()
                for value in dataset.data:
                    if np.random.random() > 0.7:
                        sketch1.add(value)
                    else:
                        sketch2.add(value)
                sketch1.merge(sketch2)
                self._evaluate_sketch_accuracy(sketch1, dataset, TEST_REL_ACC)

    def test_merge_mixed(self):
        """Test merging DDSketches of different distributions"""
        ntests = 20
        test_datasets = [Normal, Exponential, Laplace, Bimodal]
        for _ in range(ntests):
            merged_dataset = EmptyDataset(0)
            merged_sketch = self._new_dd_sketch()
            for dataset in test_datasets:
                generator = dataset(np.random.randint(0, 500))
                sketch = self._new_dd_sketch()
                for value in generator.data:
                    sketch.add(value)
                    merged_dataset.add(value)
                merged_sketch.merge(sketch)
            self._evaluate_sketch_accuracy(merged_sketch, merged_dataset, TEST_REL_ACC)

    def test_consistent_merge(self):
        """Test that merge() calls do not modify the argument sketch."""
        sketch1 = self._new_dd_sketch()
        sketch2 = self._new_dd_sketch()
        dataset = Normal(100)
        for value in dataset.data:
            sketch1.add(value)
        sketch1.merge(sketch2)
        # sketch2 is still empty
        assert sketch2.num_values == 0

        dataset = Normal(50)
        for value in dataset.data:
            sketch2.add(value)

        sketch2_summary = [sketch2.get_quantile_value(q) for q in TEST_QUANTILES] + [
            sketch2.sum,
            sketch2.avg,
            sketch2.num_values,
        ]
        sketch1.merge(sketch2)

        dataset = Normal(10)
        for value in dataset.data:
            sketch1.add(value)
        # changes to sketch1 does not affect sketch2 after merge
        sketch2_summary = [sketch2.get_quantile_value(q) for q in TEST_QUANTILES] + [
            sketch2.sum,
            sketch2.avg,
            sketch2.num_values,
        ]
        assert sketch2_summary == pytest.approx(
            [sketch2.get_quantile_value(q) for q in TEST_QUANTILES]
            + [sketch2.sum, sketch2.avg, sketch2.num_values],
        )

        sketch3 = self._new_dd_sketch()
        sketch3.merge(sketch2)
        # merging to an empty sketch does not change sketch2
        assert sketch2_summary == pytest.approx(
            [sketch2.get_quantile_value(q) for q in TEST_QUANTILES]
            + [sketch2.sum, sketch2.avg, sketch2.num_values],
        )


class TestDDSketch(BaseTestDDSketches, TestCase):
    """Class for testing LogCollapsingLowestDenseDDSketch"""

    @staticmethod
    def _new_dd_sketch():
        return DDSketch(TEST_REL_ACC)


class TestLogCollapsingLowestDenseDDSketch(BaseTestDDSketches, TestCase):
    """Class for testing LogCollapsingLowestDenseDDSketch"""

    @staticmethod
    def _new_dd_sketch():
        return LogCollapsingLowestDenseDDSketch(TEST_REL_ACC, TEST_BIN_LIMIT)


class TestLogCollapsingHighestDenseDDSketch(BaseTestDDSketches, TestCase):
    """Class for testing LogCollapsingHighestDenseDDSketch"""

    @staticmethod
    def _new_dd_sketch():
        return LogCollapsingHighestDenseDDSketch(TEST_REL_ACC, TEST_BIN_LIMIT)


def test_version():
    """Ensure the package version is exposed by the API."""
    assert hasattr(ddsketch, "__version__")
    assert isinstance(ddsketch.__version__, str)
