from collections import Counter
from math import log
import sys
import unittest

from ddsketch.store import CollapsingLowestDenseStore, DenseStore

TEST_MAX_BINS = [1, 20, 1000]
EXTREME_MAX = sys.maxsize
EXTREME_MIN = -sys.maxsize - 1


class TestStore(object):
    def test_empty(self):
        values = []
        self._test_store(values)

    def test_constant(self):
        values = [0] * 10000
        self._test_store(values)

    def test_increasing_linearly(self):
        values = list(range(10000))
        self._test_store(values)

    def test_decreasing_linearly(self):
        values = list(reversed(range(10000)))
        self._test_store(values)

    def test_increasing_exponentially(self):
        values = [2 ** x for x in range(16)]
        self._test_store(values)

    def test_decreasing_exponentially(self):
        values = [2 ** x for x in reversed(range(16))]
        self._test_store(values)

    def test_bin_counts(self):
        values = [x for x in range(10) for i in range(2 * x)]
        self._test_store(values)

        values = [-x for x in range(10) for i in range(2 * x)]
        self._test_store(values)

    def test_extreme_values(self):
        self._test_store([EXTREME_MAX])
        self._test_store([EXTREME_MIN])
        self._test_store([0, EXTREME_MIN])
        self._test_store([0, EXTREME_MAX])
        self._test_store([EXTREME_MIN, EXTREME_MAX])
        self._test_store([EXTREME_MAX, EXTREME_MIN])

    def test_merging_empty(self):
        self._test_merging([[], []])

    def test_merging_far_apart(self):
        self._test_merging([[-10000], [10000]])
        self._test_merging([[10000], [-10000]])
        self._test_merging([[10000], [-10000], [0]])
        self._test_merging([[10000, 0], [-10000], [0]])

    def test_merging_constant(self):
        self._test_merging([[2, 2], [2, 2, 2], [2]])
        self._test_merging([[-8, -8], [-8]])

    def test_merging_extreme_values(self):
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
        store = CollapsingLowestDenseStore(10)
        store.copy(CollapsingLowestDenseStore(10))
        self.assertEqual(store.count, 0)

    def test_copying_non_empty(self):
        store = CollapsingLowestDenseStore(10)
        new_store = CollapsingLowestDenseStore(10)
        new_store.add(0)
        store.copy(new_store)
        self.assertEqual(store.count, 1)


class TestDenseStore(TestStore, unittest.TestCase):
    def _test_values(self, store, values):
        counter = Counter(values)

        expected_total_count = sum(counter.values())
        self.assertEqual(expected_total_count, sum(store.bins))
        if expected_total_count == 0:
            self.assertTrue(all([x == 0 for x in store.bins]))
        else:
            self.assertFalse(all([x == 0 for x in store.bins]))

            counter = Counter(values)
            for i, b in enumerate(store.bins):
                if b != 0:
                    self.assertEqual(counter[i + store.min_key], b)

    def _test_store(self, values):
        for max_bins in TEST_MAX_BINS:
            store = DenseStore()
            for v in values:
                store.add(v)
            self._test_values(store, values)

    def _test_merging(self, list_values):
        for max_bins in TEST_MAX_BINS:
            store = DenseStore()

            for values in list_values:
                intermediate_store = DenseStore()
                for v in values:
                    intermediate_store.add(v)
                store.merge(intermediate_store)

            flat_values = [v for values in list_values for v in values]
            self._test_values(store, flat_values)

    def test_extreme_values(self):
        # DenseStore is not meant to be used with values that are extremely far from one another as it
        # would allocate an excessively large array.
        pass

    def test_merging_extreme_values(self):
        # DenseStore is not meant to be used with values that are extremely far from one another as it
        # would allocate an excessively large array.
        pass


class TestCollapsingLowestDenseStore(TestStore, unittest.TestCase):
    def _test_values(self, store, values):
        counter = Counter(values)

        expected_total_count = sum(counter.values())
        self.assertEqual(expected_total_count, sum(store.bins))
        if expected_total_count == 0:
            self.assertTrue(all([x == 0 for x in store.bins]))
        else:
            self.assertFalse(all([x == 0 for x in store.bins]))

            max_index = max(counter)
            min_storable_index = max(float("-inf"), max_index - store.max_bins + 1)
            counter = Counter([max(x, min_storable_index) for x in values])
            for i, b in enumerate(store.bins):
                if b != 0:
                    self.assertEqual(counter[i + store.min_key], b)

    def _test_store(self, values):
        for max_bins in TEST_MAX_BINS:
            store = CollapsingLowestDenseStore(max_bins)
            for v in values:
                store.add(v)
            self._test_values(store, values)

    def _test_merging(self, list_values):
        for max_bins in TEST_MAX_BINS:
            store = CollapsingLowestDenseStore(max_bins)

            for values in list_values:
                intermediate_store = CollapsingLowestDenseStore(max_bins)
                for v in values:
                    intermediate_store.add(v)
                store.merge(intermediate_store)

            flat_values = [v for values in list_values for v in values]
            self._test_values(store, flat_values)
