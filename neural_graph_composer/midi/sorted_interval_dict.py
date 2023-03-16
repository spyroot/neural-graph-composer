import collections
from neural_graph_composer.midi.Interval import Interval
from neural_graph_composer.midi.sorted_dict import SortedDict


class SortedIntervalDict(collections.abc.MutableMapping):
    """A dictionary that maps intervals to values, with interval keys sorted by their start value."""

    def __init__(self, *args, **kwargs):
        self._dict = SortedDict()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        if isinstance(key, Interval):
            return self._dict[key]
        elif isinstance(key, slice):
            start = key.start if key.start is not None else float('-inf')
            stop = key.stop if key.stop is not None else float('inf')
            return SortedIntervalDict((k, v) for k, v in self.items() if k.overlaps(Interval(start, stop)))
        else:
            raise KeyError(f"Invalid key type: {type(key)}")

    def __setitem__(self, key, value):
        if isinstance(key, Interval):
            self._dict[key] = value
        else:
            raise KeyError(f"Invalid key type: {type(key)}")

    # def __setitem__(self, key, value):
    #     if isinstance(key, Interval):
    #         i = bisect.bisect_right(self._items, (key, None))
    #         if i == 0 or self._items[i - 1][0] < key:
    #             self._items.insert(i, (key, value))
    #         else:
    #             raise ValueError("Interval overlaps with existing interval")
    #     else:
    #         raise KeyError(f"Invalid key type: {type(key)}")

    def __delitem__(self, key):
        """
        :param key:
        :return:
        """
        if isinstance(key, Interval):
            for i, (k, v) in enumerate(self._items):
                if k == key:
                    del self._items[i]
                    return
            raise KeyError(key)
        elif isinstance(key, slice):
            for k in list(self.keys())[key]:
                del self[k]
        else:
            raise KeyError(f"Invalid key type: {type(key)}")

    def __iter__(self):
        return self._dict.__iter__()

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return f"{type(self).__name__}({dict((Interval(k.start, k.end), v) for k, v in self.items())})"

    # def _remove_key_only(self, key):
    #     self._keys.remove(key)

    # def _remove_key(self, i):
    #     del self._dict[self._keys[i]]
    #     del self._keys[i]
    #     del self._values[i]
