# Unless explicitly stated otherwise all files in this repository are licensed
# under the Apache License 2.0.
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2020 Datadog, Inc.

"""Tests for DDSketch"""

from abc import ABC, abstractmethod
from collections import Counter
from unittest import TestCase

import numpy as np

from datasets import (
    Bimodal,
    Constant,
    EmptyDataset,
    Exponential,
    Integers,
    Lognormal,
    Laplace,
    Mixed,
    NegativeUniformBackward,
    NegativeUniformForward,
    NumberLineBackward,
    NumberLineForward,
    Normal,
    Trimodal,
    UniformBackward,
    UniformForward,
    UniformSqrt,
    UniformZoomIn,
    UniformZoomOut,
)
from ddsketch.ddsketch import (
    LogCollapsingHighestDenseDDSketch,
    LogCollapsingLowestDenseDDSketch,
    DDSketch,
)

test_quantiles = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1]
test_sizes = [3, 5, 10, 100, 1000]
datasets = [
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


class TestDDSketches(ABC):
    """AbstractBaseClass for testing DDSketch implementations"""

    @staticmethod
    @abstractmethod
    def _new_dd_sketch():
        """ create a new DDSketch of the appropriate type """

    def _evaluate_sketch_accuracy(self, sketch, data, eps):
        size = data.size
        for quantile in test_quantiles:
            sketch_q = sketch.get_quantile_value(quantile)
            data_q = data.quantile(quantile)
            err = abs(sketch_q - data_q)
            self.assertTrue(err - eps * abs(data_q) <= 1e-15)
        self.assertEqual(sketch.num_values, size)
        self.assertAlmostEqual(sketch.sum, data.sum)
        self.assertAlmostEqual(sketch.avg, data.avg)

    def test_distributions(self):
        """Test DDSketch on values from various distributions"""
        for dataset in datasets:
            for size in test_sizes:
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
        self.assertTrue(err - TEST_REL_ACC * abs(data_median) <= 1e-15)
        self.assertAlmostEqual(sketch.num_values, 110 * 2)
        self.assertAlmostEqual(sketch.sum, 5445 + 11000)
        self.assertAlmostEqual(sketch.avg, 74.75)

    def test_merge_equal(self):
        """Test merging equal-sized DDSketches """
        parameters = [(35, 1), (1, 3), (15, 2), (40, 0.5)]
        for size in test_sizes:
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
        """Test merging variable-sized DDSketches """
        ntests = 20
        for _ in range(ntests):
            for size in test_sizes:
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
        self.assertEqual(sketch2.num_values, 0)

        dataset = Normal(50)
        for value in dataset.data:
            sketch2.add(value)

        sketch2_summary = [sketch2.get_quantile_value(q) for q in test_quantiles] + [
            sketch2.sum,
            sketch2.avg,
            sketch2.num_values,
        ]
        sketch1.merge(sketch2)

        dataset = Normal(10)
        for value in dataset.data:
            sketch1.add(value)
        # changes to sketch1 does not affect sketch2 after merge
        sketch2_summary = [sketch2.get_quantile_value(q) for q in test_quantiles] + [
            sketch2.sum,
            sketch2.avg,
            sketch2.num_values,
        ]
        self.assertAlmostEqual(
            sketch2_summary,
            [sketch2.get_quantile_value(q) for q in test_quantiles]
            + [sketch2.sum, sketch2.avg, sketch2.num_values],
        )

        sketch3 = self._new_dd_sketch()
        sketch3.merge(sketch2)
        # merging to an empty sketch does not change sketch2
        self.assertAlmostEqual(
            sketch2_summary,
            [sketch2.get_quantile_value(q) for q in test_quantiles]
            + [sketch2.sum, sketch2.avg, sketch2.num_values],
        )


class TestDDSketch(TestDDSketches, TestCase):
    """Class for testing LogCollapsingLowestDenseDDSketch"""

    @staticmethod
    def _new_dd_sketch():
        return DDSketch(TEST_REL_ACC)


class TestLogCollapsingLowestDenseDDSketch(TestDDSketches, TestCase):
    """Class for testing LogCollapsingLowestDenseDDSketch"""

    @staticmethod
    def _new_dd_sketch():
        return LogCollapsingLowestDenseDDSketch(TEST_REL_ACC, TEST_BIN_LIMIT)


class TestLogCollapsingHighestDenseDDSketch(TestDDSketches, TestCase):
    """Class for testing LogCollapsingHighestDenseDDSketch"""

    @staticmethod
    def _new_dd_sketch():
        return LogCollapsingHighestDenseDDSketch(TEST_REL_ACC, TEST_BIN_LIMIT)
