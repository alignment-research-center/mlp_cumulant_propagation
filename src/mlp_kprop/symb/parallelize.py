import hashlib

try:
    from mpi4py import MPI

    is_parallel = MPI.COMM_WORLD.size > 1
    is_root = MPI.COMM_WORLD.rank == 0
except ImportError:
    is_parallel = False
    is_root = True


def multi_map_serial(func, iterable):
    def generator():
        for item in iterable:
            yield from func(*item)

    return generator()


class MultiMap:
    def __init__(self, source):
        self.source = source
        self.funcs = []
        self.comm = MPI.COMM_WORLD

    def append_func(self, func):
        self.funcs.append(func)

    def index_rank(self, index):
        digest = hashlib.sha256(str(index).encode()).digest()
        return int.from_bytes(digest[:8], byteorder="big") % self.comm.size

    def local_iter(self):
        index = -1
        for index, item in enumerate(self.source):
            if self.index_rank(index) == self.comm.rank:
                items = [item]
                for func in self.funcs:
                    items = [
                        new_item for old_item in items for new_item in func(*old_item)
                    ]
                yield items
        self.source_length = index + 1

    def __iter__(self):
        local_items = list(self.local_iter())
        global_items = self.comm.allgather(local_items)
        global_items = [iter(items) for items in global_items]
        for index in range(self.source_length):
            index_rank = self.index_rank(index)
            yield from next(global_items[index_rank])


def multi_map_parallel(func, iterable):
    if not isinstance(iterable, MultiMap):
        iterable = MultiMap(iterable)
    iterable.append_func(func)
    return iterable


multi_map = multi_map_parallel if is_parallel else multi_map_serial
