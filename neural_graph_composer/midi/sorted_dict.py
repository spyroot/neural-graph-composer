"""
A dictionary that maintains the keys sorted order.


"""
import collections
import bisect


class SortedDict(collections.OrderedDict):
    """
    A dictionary that maintains the keys sorted.

    This class extends the `OrderedDict` class from the `collections` module
    and maintains the keys sorted in ascending order. It provides methods to
    access the keys, values, and items in sorted order.

    Examples:
        >>> sd = SortedDict({3: 'c', 1: 'a', 2: 'b'})
        >>> sd.keys()
        SortedKeysView(SortedDict({1: 'a', 2: 'b', 3: 'c'}))
        >>> sd.values()
        ['a', 'b', 'c']
        >>> sd.items()
        [(1, 'a'), (2, 'b'), (3, 'c')]

    Insertion: O(log n)
    Deletion: O(log n)
    Lookup: O(1)
    Traversal: O(n)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._keys = sorted(super().keys())

    def __setitem__(self, key, value):
        """Set the value of a key.
        If the key is already in the dictionary, remove it first to maintain
        the sorted order. Insert the key in the sorted position using binary
        search.
        :param key:
        :param value:
        :return:
        """
        if key in self:
            del self[key]

        pos = bisect.bisect_left(self._keys, key)
        self._keys.insert(pos, key)
        super().__setitem__(key, value)

    def __iter__(self):
        """Return an iterator over the keys sorted in ascending order."""
        return iter(self.keys())

    def __delitem__(self, key):
        """Delete a key and its value.
        :param key: The key to delete
        :return:
        """
        super().__delitem__(key)
        self._keys.remove(key)

    def keys(self):
        """Return an iterator over the keys sorted in ascending order.
        :return: A sorted iterator over the keys
        """
        return self._keys

    def values(self):
        """Return a list of values corresponding to the keys sorted in ascending order.
        :return: A list of values corresponding to the sorted keys.
        """
        return [self[key] for key in self.keys()]

    def items(self):
        """ Return an iterator over the (key, value) pairs sorted in ascending order.
        :return:
        """
        return [(key, self[key]) for key in self.keys()]

    def __reversed__(self):
        """Return a reverse iterator over the keys in the dictionary.
        :return: A reverse iterator over the keys.
        """
        return reversed(self._keys)

    #
    # def __getitem__(self, item):
    #     """Get the value for a key or an interval of keys in the dictionary.
    #     :param item: The key or interval of keys to get.
    #     :return: The value for the key or keys.
    #     """
    #     if isinstance(item, slice):
    #         keys = self.keys()[item]
    #         return SortedDict((k, self[k]) for k in keys)
    #     else:
    #         return super().__getitem__(item)
