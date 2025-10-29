"""Provides simple extension containers and analysis."""

from collections import Counter
import itertools


class SortedSet(set):
    """A set that maintains strict sorting when iterated upon."""
    def __init__(self, *args, sorted=sorted, **kwargs):
        # TODO(eric): Add `strict` option to prevent duplicates.
        self._sorted = sorted
        super().__init__(*args, **kwargs)

    def __iter__(self):
        original = super().__iter__()
        return iter(self._sorted(original))


class SortedDict(dict):
    """A dict that maintains strict sorting when iterated upon.
    All values are returned according to the sorting of keys, not the
    values."""
    def __init__(self, *args, sorted_keys=sorted, **kwargs):
        # TODO(eric): Add `strict` option to prevent duplicates.
        self._sorted_keys = sorted_keys
        super().__init__(*args, **kwargs)

    def __iter__(self):
        original = super().__iter__()
        return iter(self._sorted_keys(original))

    def __repr__(self):
        items_str = ", ".join(f"{repr(k)}: {repr(v)}" for k, v in self.items())
        return f"SortedDict({{{items_str}}})"

    def items(self):
        out = []
        for key in self:
            out.append((key, self[key]))
        return out

    def keys(self):
        return list(iter(self))

    def values(self):
        items = self.items()
        return [value for (_, value) in items]


def dict_items_zip(*items):
    """
    Provides `zip()`-like functionality for the items of a list of
    dictionaries. This requires that all dictionaries have the same keys
    (though possibly in a different order).

    Returns:
        Iterable[key, values], where ``values`` is a tuple of the value from
        each dictionary.
    """
    if len(items) == 0:
        # Return an empty iterator.
        return
    first = items[0]
    assert isinstance(first, dict)
    check_keys = set(first.keys())
    for item in items[1:]:
        assert isinstance(item, dict)
        assert set(item.keys()) == check_keys
    for k in first.keys():
        values = tuple(item[k] for item in items)
        yield k, values


def take_first(iterable):
    """
    Robustly gets the first item from an iterable and returns it.
    You should always use this isntead of `next(iter(...))`; e.g. instead of

        my_first = next(iter(container))

    you should instead do:

        my_first = take_first(container)
    """
    first, = itertools.islice(iterable, 1)
    return first


def exclusive_dict_update(dest, src):
    common = set(dest.keys()) & set(src.keys())
    assert len(common) == 0, f"Non-exclusive keys: {common}"
    dest.update(src)
    return dest


def assert_unique(xs):
    assert isinstance(xs, list)
    bad = {value: count for value, count in Counter(xs).items() if count > 1}
    # Print all duplicates.
    assert len(bad) == 0, f"Duplicates {{key: count}}:\n{bad}"


_MARKER = object()


def interleave_longest(iters):
    """
    Interleaves elements in each iterator in ``iters``, which may produce
    sequences of different lengths.

    Derived from:'
    - https://stackoverflow.com/a/40954220/7829525
    - https://more-itertools.readthedocs.io/en/v8.12.0/_modules/more_itertools/more.html#interleave_longest
    """  # noqa
    zip_iter = itertools.zip_longest(*iters, fillvalue=_MARKER)
    for x in itertools.chain(*zip_iter):
        if x is not _MARKER:
            yield x
