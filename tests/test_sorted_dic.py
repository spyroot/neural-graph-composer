import unittest

from neural_graph_composer.midi.sorted_dict import SortedDict


class TestSortedDict(unittest.TestCase):

    def test_empty_dict(self):
        # Test that an empty dictionary is created and that it has no items.
        sorted_dict = SortedDict()
        self.assertEqual(len(sorted_dict), 0)
        self.assertEqual(list(sorted_dict.keys()), [])
        self.assertEqual(list(sorted_dict.values()), [])
        self.assertEqual(list(sorted_dict.items()), [])

    def test_insertion_order(self):
        sd = SortedDict()
        sd[1] = 'a'
        sd[3] = 'c'
        sd[2] = 'b'

        self.assertEqual(list(sd.keys()), [1, 2, 3])
        self.assertEqual(list(sd.values()), ['a', 'b', 'c'])

    def test_update(self):
        """ need fix this
        :return:
        """
        sd = SortedDict({1: 'a', 2: 'b', 3: 'c'})
        sd.update({4: 'd', 5: 'e'})
        # self.assertEqual(sd.keys(), SortedKeysView(SortedDict({1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e'})))

    def test_reverse_sorting(self):
        """Test that the SortedDict maintains its order when reversed.
        :return:
        """
        sd = SortedDict({3: 'c', 1: 'a', 2: 'b', 4: 'd'})
        self.assertEqual(list(sd.keys()), [1, 2, 3, 4])
        self.assertEqual(list(reversed(sd.keys())), [4, 3, 2, 1])
        self.assertEqual(list(reversed(sd.values())), ['d', 'c', 'b', 'a'])
        self.assertEqual(list(reversed(sd.items())), [(4, 'd'), (3, 'c'), (2, 'b'), (1, 'a')])

    # def test_interval_keys(self):
    #     sd = SortedDict({(0, 5): 'a', (5, 10): 'b', (10, 15): 'c', (15, 20): 'd'})
    #     self.assertEqual(list(sd.keys()), [(0, 5), (5, 10), (10, 15), (15, 20)])
    #     self.assertEqual(list(sd.values()), ['a', 'b', 'c', 'd'])
    #     self.assertEqual(sd[(0, 5)], 'a')
    #     self.assertEqual(sd[(5, 10)], 'b')
    #     self.assertEqual(sd[(10, 15)], 'c')
    #     self.assertEqual(sd[(15, 20)], 'd')
    #     self.assertEqual(sd[0:5], SortedDict({(0, 5): 'a'}))
    #     self.assertEqual(sd[5:10], SortedDict({(5, 10): 'b'}))
    #     self.assertEqual(sd[10:15], SortedDict({(10, 15): 'c'}))
    #     self.assertEqual(sd[15:20], SortedDict({(15, 20): 'd'}))
    #     self.assertEqual(sd[5:], SortedDict({(5, 10): 'b', (10, 15): 'c', (15, 20): 'd'}))
    #     self.assertEqual(sd[:15], SortedDict({(0, 5): 'a', (5, 10): 'b', (10, 15): 'c'}))
    #     self.assertEqual(sd[:], sd)

    # def test_getitem_slice(self):
    #     sd = SortedDict({3: 'c', 1: 'a', 2: 'b', 4: 'd', 0: 'e'})
    #     self.assertEqual(sd[1:4], SortedDict({1: 'a', 2: 'b', 3: 'c'}))
    #     self.assertEqual(sd[1:], SortedDict({1: 'a', 2: 'b', 3: 'c', 4: 'd', 0: 'e'}))
    #     self.assertEqual(sd[:3], SortedDict({0: 'e', 1: 'a', 2: 'b', 3: 'c'}))
    #     self.assertEqual(sd[:], sd)
