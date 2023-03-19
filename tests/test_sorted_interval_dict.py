from neural_graph_composer.midi.Interval import Interval
# from neural_graph_composer.midi.sorted_interval_dict import SortedIntervalDict
import unittest
from neural_graph_composer.midi.sorted_interval_dict import SortedIntervalDict


class TestSortedIntervalDict(unittest.TestCase):

    def test_insertion_order(self):
        sid = SortedIntervalDict()
        sid[Interval(1, 3)] = 'a'
        sid[Interval(2, 4)] = 'b'
        sid[Interval(3, 5)] = 'c'
        self.assertEqual(list(sid.keys()), [Interval(1, 3), Interval(2, 4), Interval(3, 5)])

    # def test_update(self):
    #     sid = SortedIntervalDict()
    #     sid[Interval(1, 3)] = 'a'
    #     sid[Interval(2, 4)] = 'b'
    #     sid[Interval(3, 5)] = 'c'
    #     sid.update({Interval(4, 6): 'd', Interval(5, 7): 'e'})
    #     self.assertEqual(list(sid.keys()), [Interval(1, 3), Interval(2, 4), Interval(3, 5), Interval(4, 6), Interval(5, 7)])
    #     self.assertEqual(list(sid.values()), ['a', 'b', 'c', 'd', 'e'])
    #
    # def test_delitem(self):
    #     sid = SortedIntervalDict()
    #     sid[Interval(1, 3)] = 'a'
    #     sid[Interval(2, 4)] = 'b'
    #     sid[Interval(3, 5)] = 'c'
    #     del sid[Interval(2, 4)]
    #     self.assertEqual(list(sid.keys()), [Interval(1, 3), Interval(3, 5)])
    #     self.assertEqual(list(sid.values()), ['a', 'c'])
