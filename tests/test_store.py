# Unless explicitly stated otherwise all files in this repository are licensed
# under the Apache License 2.0.
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2020 Datadog, Inc.

"""Tests for the Store classes"""

import abc
from collections import Counter
import sys
from unittest import TestCase

import six

from ddsketch.store import CollapsingHighestDenseStore
from ddsketch.store import CollapsingLowestDenseStore
from ddsketch.store import DenseStore


TEST_BIN_LIMIT = [1, 20, 1000]
EXTREME_MAX = sys.maxsize
EXTREME_MIN = -sys.maxsize - 1


class BaseTestStore(six.with_metaclass(abc.ABCMeta)):
    """Base class for testing Store classes"""

    @abc.abstractmethod
    def _test_values(self, store, values):
        """Test the store's bin counts against what we expect"""

    @abc.abstractmethod
    def _test_store(self, values):
        """Initialize the store; add the values; call _test_values"""

    @abc.abstractmethod
    def _test_merging(self, list_values):
        """
        Initialize the stores; for each values in list_values, add them to the
        corresponding store; merge the stores; test the merged store's bin
        counts against what we expect.
        """

    def test_empty(self):
        """Test no values"""
        values = []
        self._test_store(values)

    def test_constant(self):
        """Test a constant stream of values"""
        values = [0] * 10000
        self._test_store(values)

    def test_increasing_linearly(self):
        """Test a stream of increasing values"""
        values = list(range(10000))
        self._test_store(values)

    def test_decreasing_linearly(self):
        """Test a stream of decreasing values"""
        values = list(reversed(range(10000)))
        self._test_store(values)

    def test_increasing_exponentially(self):
        """Test a stream of values increasing exponentially"""
        values = [2 ** x for x in range(16)]
        self._test_store(values)

    def test_decreasing_exponentially(self):
        """Test a stream of values decreasing exponentially"""
        values = [2 ** x for x in reversed(range(16))]
        self._test_store(values)

    def test_bin_counts(self):
        """Test bin counts for positive and negative numbers"""
        values = [x for x in range(10) for i in range(2 * x)]
        self._test_store(values)

        values = [-x for x in range(10) for i in range(2 * x)]
        self._test_store(values)

    def test_extreme_values(self):
        """Test extreme values"""
        self._test_store([EXTREME_MAX])
        self._test_store([EXTREME_MIN])
        self._test_store([0, EXTREME_MIN])
        self._test_store([0, EXTREME_MAX])
        self._test_store([EXTREME_MIN, EXTREME_MAX])
        self._test_store([EXTREME_MAX, EXTREME_MIN])

    def test_merging_empty(self):
        """Test merging empty stores"""
        self._test_merging([[], []])

    def test_merging_far_apart(self):
        """Test merging stores with values that are far apart"""
        self._test_merging([[-10000], [10000]])
        self._test_merging([[10000], [-10000]])
        self._test_merging([[10000], [-10000], [0]])
        self._test_merging([[10000, 0], [-10000], [0]])

    def test_merging_constant(self):
        """Test merging stores with the same constants"""
        self._test_merging([[2, 2], [2, 2, 2], [2]])
        self._test_merging([[-8, -8], [-8]])

    def test_merging_extreme_values(self):
        """Test merging stores with extreme values"""
        self._test_merging([[0], [EXTREME_MIN]])
        self._test_merging([[0], [EXTREME_MAX]])
        self._test_merging([[EXTREME_MIN], [0]])
        self._test_merging([[EXTREME_MAX], [0]])
        self._test_merging([[EXTREME_MIN], [EXTREME_MIN]])
        self._test_merging([[EXTREME_MAX], [EXTREME_MAX]])
        self._test_merging([[EXTREME_MIN], [EXTREME_MAX]])
        self._test_merging([[EXTREME_MAX], [EXTREME_MIN]])
        self._test_merging([[0], [EXTREME_MIN, EXTREME_MAX]])
        self._test_merging([[EXTREME_MIN, EXTREME_MAX], [0]])

    def test_copying_empty(self):
        """Test copying empty stores"""
        store = CollapsingLowestDenseStore(10)
        store.copy(CollapsingLowestDenseStore(10))
        assert store.count == 0

    def test_copying_non_empty(self):
        """Test copying stores"""
        store = CollapsingLowestDenseStore(10)
        new_store = CollapsingLowestDenseStore(10)
        new_store.add(0)
        store.copy(new_store)
        assert store.count == 1


class TestDenseStore(BaseTestStore, TestCase):
    """Class for testing the DenseStore class"""

    def _test_values(self, store, values):
        counter = Counter(values)

        expected_total_count = sum(counter.values())
        assert expected_total_count == sum(store.bins)
        if expected_total_count == 0:
            assert all([x == 0 for x in store.bins])
        else:
            assert not all([x == 0 for x in store.bins])

            counter = Counter(values)
            for i, sbin in enumerate(store.bins):
                if sbin != 0:
                    assert counter[i + store.offset] == sbin

    def _test_store(self, values):
        store = DenseStore()
        for val in values:
            store.add(val)
        self._test_values(store, values)

    def _test_merging(self, list_values):
        store = DenseStore()

        for values in list_values:
            intermediate_store = DenseStore()
            for val in values:
                intermediate_store.add(val)
            store.merge(intermediate_store)

        flat_values = [v for values in list_values for v in values]
        self._test_values(store, flat_values)

    def test_key_at_rank(self):
        """Test that key_at_rank properly handles decimal ranks"""
        store = DenseStore()
        store.add(4)
        store.add(10)
        store.add(100)
        assert store.key_at_rank(0) == 4
        assert store.key_at_rank(1) == 10
        assert store.key_at_rank(2) == 100
        assert store.key_at_rank(0, lower=False) == 4
        assert store.key_at_rank(1, lower=False) == 10
        assert store.key_at_rank(2, lower=False) == 100
        assert store.key_at_rank(0.5) == 4
        assert store.key_at_rank(1.5) == 10
        assert store.key_at_rank(2.5) == 100
        assert store.key_at_rank(-0.5, lower=False) == 4
        assert store.key_at_rank(0.5, lower=False) == 10
        assert store.key_at_rank(1.5, lower=False) == 100

    def test_extreme_values(self):
        """Override. DenseStore is not meant to be used with values that are extremely
        far from one another as it would allocate an excessively large
        array.
        """

    def test_merging_extreme_values(self):
        """Override. DenseStore is not meant to be used with values that are extremely
        far from one another as it would allocate an excessively large
        array.
        """


class TestCollapsingLowestDenseStore(BaseTestStore, TestCase):
    """Class for testing the CollapsingLowestDenseStore class"""

    def _test_values(self, store, values):
        counter = Counter(values)
        expected_total_count = sum(counter.values())
        assert expected_total_count == sum(store.bins)

        if expected_total_count == 0:
            assert all([x == 0 for x in store.bins])
        else:
            assert not all([x == 0 for x in store.bins])

            max_index = max(counter)
            min_storable_index = max(float("-inf"), max_index - store.bin_limit + 1)
            counter = Counter([max(x, min_storable_index) for x in values])

            for i, sbin in enumerate(store.bins):
                if sbin != 0:
                    assert counter[i + store.offset] == sbin

    def _test_store(self, values):
        for bin_limit in TEST_BIN_LIMIT:
            store = CollapsingLowestDenseStore(bin_limit)
            for val in values:
                store.add(val)
            self._test_values(store, values)

    def _test_merging(self, list_values):
        for bin_limit in TEST_BIN_LIMIT:
            store = CollapsingLowestDenseStore(bin_limit)

            for values in list_values:
                intermediate_store = CollapsingLowestDenseStore(bin_limit)
                for val in values:
                    intermediate_store.add(val)
                store.merge(intermediate_store)
            flat_values = [v for values in list_values for v in values]
            self._test_values(store, flat_values)


class TestCollapsingHighestDenseStore(BaseTestStore, TestCase):
    """Class for testing the CollapsingHighestDenseStore class"""

    def _test_values(self, store, values):
        counter = Counter(values)

        expected_total_count = sum(counter.values())
        assert expected_total_count == sum(store.bins)
        if expected_total_count == 0:
            assert all([x == 0 for x in store.bins])
        else:
            assert not all([x == 0 for x in store.bins])

            min_index = min(counter)
            max_storable_index = min(float("+inf"), min_index + store.bin_limit - 1)
            counter = Counter([min(x, max_storable_index) for x in values])

            for i, sbin in enumerate(store.bins):
                if sbin != 0:
                    assert counter[i + store.offset] == sbin

    def _test_store(self, values):
        for bin_limit in TEST_BIN_LIMIT[1:2]:
            store = CollapsingHighestDenseStore(bin_limit)
            for val in values:
                store.add(val)
            self._test_values(store, values)

    def _test_merging(self, list_values):
        for bin_limit in TEST_BIN_LIMIT:
            store = CollapsingHighestDenseStore(bin_limit)

            for values in list_values:
                intermediate_store = CollapsingHighestDenseStore(bin_limit)
                for val in values:
                    intermediate_store.add(val)
                store.merge(intermediate_store)
            flat_values = [v for values in list_values for v in values]
            self._test_values(store, flat_values)
