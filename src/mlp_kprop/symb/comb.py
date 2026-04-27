import itertools


class NonConsuming:
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        self.iterable, self.consuming = itertools.tee(self.iterable, 2)
        return iter(self.consuming)


def non_consuming_product(*iterables, repeat=1):
    """
    Like itertools.product, but doesn't immediately consume iterables.

    Example:
    >>> list(itertools.islice(non_consuming_product(*(itertools.count() for _ in range(3))), 5))
    [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4)]
    """
    iterables = [
        it if isinstance(it, NonConsuming) else NonConsuming(it) for it in iterables
    ] * repeat
    if not iterables:
        yield ()
        return
    first, *rest = iterables
    for item in first:
        for rest_items in non_consuming_product(*rest):
            yield (item,) + rest_items


def partitions(items, banned=()):
    items = list(items)
    if not items:
        yield []
        return
    first, *rest = items
    for part in partitions(rest, banned=banned):
        for i in range(len(part)):
            if any(
                (first, other) in banned or (other, first) in banned
                for other in part[i]
            ):
                continue
            yield [
                (*((first,) if j == i else ()), *block) for j, block in enumerate(part)
            ]
        yield [(first,), *part]


def pairings(items, banned=()):
    items = list(items)
    if len(items) == 0:
        yield []
        return
    first, *rest = items
    for i, item in enumerate(rest):
        if (first, item) in banned or (item, first) in banned:
            continue
        first_pair = (first, item)
        for pairing in pairings(rest[:i] + rest[i + 1 :], banned):
            yield [first_pair] + pairing
