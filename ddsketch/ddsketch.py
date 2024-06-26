# Unless explicitly stated otherwise all files in this repository are licensed
# under the Apache License 2.0.
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2020 Datadog, Inc.

"""A quantile sketch with relative-error guarantees. This sketch computes
quantile values with an approximation error that is relative to the actual
quantile value. It works on both negative and non-negative input values.

For instance, using DDSketch with a relative accuracy guarantee set to 1%, if
the expected quantile value is 100, the computed quantile value is guaranteed to
be between 99 and 101. If the expected quantile value is 1000, the computed
quantile value is guaranteed to be between 990 and 1010.

DDSketch works by mapping floating-point input values to bins and counting the
number of values for each bin. The underlying structure that keeps track of bin
counts is store.

The memory size of the sketch depends on the range that is covered by the input
values: the larger that range, the more bins are needed to keep track of the
input values. As a rough estimate, if working on durations with a relative
accuracy of 2%, about 2kB (275 bins) are needed to cover values between 1
millisecond and 1 minute, and about 6kB (802 bins) to cover values between 1
nanosecond and 1 day.

The size of the sketch can be have a fail-safe upper-bound by using collapsing
stores. As shown in
<a href="http://www.vldb.org/pvldb/vol12/p2195-masson.pdf">the DDSketch paper</a>
the likelihood of a store collapsing when using the default bound is vanishingly
small for most data.

DDSketch implementations are also available in:
<a href="https://github.com/DataDog/sketches-go/">Go</a>
<a href="https://github.com/DataDog/sketches-py/">Python</a>
<a href="https://github.com/DataDog/sketches-js/">JavaScript</a>
"""
import typing

from .mapping import LogarithmicMapping
from .store import CollapsingHighestDenseStore
from .store import CollapsingLowestDenseStore
from .store import DenseStore


if typing.TYPE_CHECKING:
    from typing import Optional  # noqa: F401

    from .mapping import KeyMapping  # noqa: F401
    from .store import Store  # noqa: F401


DEFAULT_REL_ACC = 0.01  # "alpha" in the paper
DEFAULT_BIN_LIMIT = 2048


class BaseDDSketch(object):
    """The base implementation of DDSketch with neither mapping nor storage specified.

    Args:
        mapping (mapping.KeyMapping): map btw values and store bins
        store (store.Store): storage for positive values
        negative_store (store.Store): storage for negative values
        zero_count (float): The count of zero values

    Attributes:
        relative_accuracy (float): the accuracy guarantee; referred to as alpha
            in the paper. (0. < alpha < 1.)

        count: the number of values seen by the sketch
        min: the minimum value seen by the sketch
        max: the maximum value seen by the sketch
        sum: the sum of the values seen by the sketch
    """

    def __init__(
        self,
        mapping,
        store,
        negative_store,
        zero_count,
    ):
        # type: (KeyMapping, Store, Store, float) -> None
        self._mapping = mapping
        self._store = store
        self._negative_store = negative_store
        self._zero_count = zero_count

        self._relative_accuracy = mapping.relative_accuracy
        self._count = self._negative_store.count + self._zero_count + self._store.count
        self._min = float("+inf")
        self._max = float("-inf")
        self._sum = 0.0

    def __repr__(self):
        # type: () -> str
        return (
            "store: {}, negative_store: {}, "
            "zero_count: {}, count: {}, "
            "sum: {}, min: {}, max: {}"
        ).format(
            self._store,
            self._negative_store,
            self._zero_count,
            self._count,
            self._sum,
            self._min,
            self._max,
        )

    @property
    def count(self):
        return self._count

    @property
    def name(self):
        # type: () -> str
        """str: name of the sketch"""
        return "DDSketch"

    @property
    def num_values(self):
        # type: () -> float
        """Return the number of values in the sketch."""
        return self._count

    @property
    def avg(self):
        # type: () -> float
        """Return the exact average of the values added to the sketch."""
        return self._sum / self._count

    @property
    def sum(self):  # noqa: A003
        # type: () -> float
        """Return the exact sum of the values added to the sketch."""
        return self._sum

    def add(self, val, weight=1.0):
        # type: (float, float) -> None
        """Add a value to the sketch."""
        if weight <= 0.0:
            raise ValueError("weight must be a positive float, got %r" % weight)

        if val > self._mapping.min_possible:
            self._store.add(self._mapping.key(val), weight)
        elif val < -self._mapping.min_possible:
            self._negative_store.add(self._mapping.key(-val), weight)
        else:
            self._zero_count += weight

        # Keep track of summary stats
        self._count += weight
        self._sum += val * weight
        if val < self._min:
            self._min = val
        if val > self._max:
            self._max = val

    def get_quantile_value(self, quantile):
        # type: (float) -> Optional[float]
        """Return the approximate value at the specified quantile.

        Args:
            quantile (float): 0 <= q <=1

        Returns:
            the value at the specified quantile or None if the sketch is empty
        """
        if quantile < 0 or quantile > 1 or self._count == 0:
            return None

        rank = quantile * (self._count - 1)
        if rank < self._negative_store.count:
            reversed_rank = self._negative_store.count - rank - 1
            key = self._negative_store.key_at_rank(reversed_rank, lower=False)
            quantile_value = -self._mapping.value(key)
        elif rank < self._zero_count + self._negative_store.count:
            return 0
        else:
            key = self._store.key_at_rank(
                rank - self._zero_count - self._negative_store.count
            )
            quantile_value = self._mapping.value(key)
        return quantile_value

    def merge(self, sketch):
        # type: (BaseDDSketch) -> None
        """Merge the given sketch into this one. After this operation, this sketch
        encodes the values that were added to both this and the input sketch.
        """
        if not self._mergeable(sketch):
            raise ValueError(
                "Cannot merge two DDSketches with different parameters, got %r and %r"
                % (self._mapping.gamma, sketch._mapping.gamma)
            )

        if sketch.count == 0:
            return

        if self._count == 0:
            self._copy(sketch)
            return

        # Merge the stores
        self._store.merge(sketch._store)
        self._negative_store.merge(sketch._negative_store)
        self._zero_count += sketch._zero_count

        # Merge summary stats
        self._count += sketch._count
        self._sum += sketch._sum
        if sketch._min < self._min:
            self._min = sketch._min
        if sketch._max > self._max:
            self._max = sketch._max

    def _mergeable(self, other):
        # type: (BaseDDSketch) -> bool
        """Two sketches can be merged only if their gammas are equal."""
        return self._mapping.gamma == other._mapping.gamma

    def _copy(self, sketch):
        # type: (BaseDDSketch) -> None
        """Copy the input sketch into this one"""
        self._store.copy(sketch._store)
        self._negative_store.copy(sketch._negative_store)
        self._zero_count = sketch._zero_count
        self._min = sketch._min
        self._max = sketch._max
        self._count = sketch._count
        self._sum = sketch._sum


class DDSketch(BaseDDSketch):
    """The default implementation of BaseDDSketch, with optimized memory usage at
    the cost of lower ingestion speed, using an unlimited number of bins. The
    number of bins will not exceed a reasonable number unless the data is
    distributed with tails heavier than any subexponential.
    (cf. http://www.vldb.org/pvldb/vol12/p2195-masson.pdf)
    """

    def __init__(self, relative_accuracy=None):
        # type: (Optional[float]) -> None
        # Make sure the parameters are valid
        if relative_accuracy is None:
            relative_accuracy = DEFAULT_REL_ACC

        mapping = LogarithmicMapping(relative_accuracy)
        store = DenseStore()
        negative_store = DenseStore()
        super(DDSketch, self).__init__(
            mapping=mapping,
            store=store,
            negative_store=negative_store,
            zero_count=0.0,
        )


class LogCollapsingLowestDenseDDSketch(BaseDDSketch):
    """Implementation of BaseDDSketch with optimized memory usage at the cost of
    lower ingestion speed, using a limited number of bins. When the maximum
    number of bins is reached, bins with lowest indices are collapsed, which
    causes the relative accuracy to be lost on the lowest quantiles. For the
    default bin limit, collapsing is unlikely to occur unless the data is
    distributed with tails heavier than any subexponential.
    (cf. http://www.vldb.org/pvldb/vol12/p2195-masson.pdf)
    """

    def __init__(self, relative_accuracy=None, bin_limit=None):
        # type: (Optional[float], Optional[int]) -> None
        # Make sure the parameters are valid
        if relative_accuracy is None:
            relative_accuracy = DEFAULT_REL_ACC

        if bin_limit is None or bin_limit < 0:
            bin_limit = DEFAULT_BIN_LIMIT

        mapping = LogarithmicMapping(relative_accuracy)
        store = CollapsingLowestDenseStore(bin_limit)
        negative_store = CollapsingLowestDenseStore(bin_limit)
        super(LogCollapsingLowestDenseDDSketch, self).__init__(
            mapping=mapping,
            store=store,
            negative_store=negative_store,
            zero_count=0.0,
        )


class LogCollapsingHighestDenseDDSketch(BaseDDSketch):
    """Implementation of BaseDDSketch with optimized memory usage at the cost of
    lower ingestion speed, using a limited number of bins. When the maximum
    number of bins is reached, bins with highest indices are collapsed, which
    causes the relative accuracy to be lost on the highest quantiles. For the
    default bin limit, collapsing is unlikely to occur unless the data is
    distributed with tails heavier than any subexponential.
    (cf. http://www.vldb.org/pvldb/vol12/p2195-masson.pdf)
    """

    def __init__(self, relative_accuracy=None, bin_limit=None):
        # type: (Optional[float], Optional[int]) -> None
        # Make sure the parameters are valid
        if relative_accuracy is None:
            relative_accuracy = DEFAULT_REL_ACC

        if bin_limit is None or bin_limit < 0:
            bin_limit = DEFAULT_BIN_LIMIT

        mapping = LogarithmicMapping(relative_accuracy)
        store = CollapsingHighestDenseStore(bin_limit)
        negative_store = CollapsingHighestDenseStore(bin_limit)
        super(LogCollapsingHighestDenseDDSketch, self).__init__(
            mapping=mapping,
            store=store,
            negative_store=negative_store,
            zero_count=0.0,
        )
