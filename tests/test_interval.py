"""
Unit test for intervals
Note I need properly check float and numpy and some other edge cases.

Author Mus spyroot@gmail.com
"""

from neural_graph_composer.midi.Interval import Interval
import unittest


class TestSortedIntervalDict(unittest.TestCase):
    def test_set(self):
        """

        :return:
        """
        s = set()
        i1 = Interval(0, 4)
        i2 = Interval(2, 5)
        s.add(i1)
        self.assertTrue(i1 in s)
        self.assertTrue(i2 not in s)
        s.add(i2)
        self.assertTrue(i2 in s)

    def test_hash(self):
        """

        :return:
        """
        i1 = Interval(0, 4)
        i2 = Interval(2, 5)
        d = {i1: "value1", i2: "value2"}
        self.assertEqual(d[i1], "value1")
        self.assertEqual(d[i2], "value2")

    def test_intersection(self):
        """
        :return:
        """
        i1 = Interval(0, 4)
        i2 = Interval(2, 5)
        self.assertEqual(i1.intersection(i2), Interval(2, 4))
        self.assertEqual(i2.intersection(i1), Interval(2, 4))
        i3 = Interval(6, 7)
        self.assertIsNone(i1.intersection(i3))
        self.assertIsNone(i3.intersection(i1))

    def test_interval_comparison(self):
        """
        :return:
        """
        i1 = Interval(1, 3)
        i2 = Interval(2, 4)
        i3 = Interval(1, 4)
        i4 = Interval(4, 6)
        i5 = Interval(2, 3)
        i6 = Interval(3, 5)

        # Test equality
        self.assertEqual(i1, Interval(1, 3))
        self.assertNotEqual(i1, i2)
        self.assertNotEqual(i1, i4)

        # Test less than and greater than
        self.assertTrue(i1 < i2)
        self.assertTrue(i2 > i1)
        self.assertTrue(i1 < i4)
        self.assertTrue(i4 > i1)

        # Test less than or equal to and greater than or equal to
        self.assertTrue(i1 <= i2)
        self.assertTrue(i2 >= i1)
        self.assertTrue(i1 <= i4)
        self.assertTrue(i4 >= i1)
        self.assertTrue(i1 <= i3)
        self.assertTrue(i3 >= i1)

        # Test overlapping intervals are not equal
        self.assertNotEqual(i1, i5)
        self.assertNotEqual(i1, i6)
        self.assertNotEqual(i2, i5)
        self.assertNotEqual(i2, i6)

        # Test non-overlapping intervals are not equal
        self.assertNotEqual(i1, i4)
        self.assertNotEqual(i2, i4)
        self.assertNotEqual(i3, i4)

    def test_interval_edge_cases(self):
        # Test when start and end values are equal
        i = Interval(3, 3)
        #
        self.assertEqual(i.start, 3)
        self.assertEqual(i.end, 3)
        self.assertTrue(3 in i)
        self.assertEqual(hash(i), hash(Interval(3, 3)))
        self.assertEqual(str(i), "[3, 3]")
        self.assertEqual(repr(i), "[3, 3]")
        #
        self.assertTrue(Interval(2, 3) < i)
        self.assertTrue(i > Interval(2, 3))

        self.assertTrue(Interval(3, 4) > i)
        self.assertTrue(i < Interval(3, 4))

        self.assertTrue(Interval(2, 3) <= i)
        self.assertFalse(i <= Interval(2, 3))
        self.assertTrue(i > Interval(2, 3))

        # self.assertFalse(i <= Interval(2, 3))
        # self.assertTrue(i <= Interval(3, 3))
        # self.assertFalse(i <= Interval(3, 4))
        self.assertTrue(i <= Interval(4, 5))
        self.assertTrue(Interval(2, 3) <= i <= Interval(4, 5))

        self.assertTrue(i, i.intersection(Interval(2, 3)))
        self.assertTrue(i, i.intersection(Interval(3, 4)))

    def test_interval_negative_values(self):
        # Test an interval with negative values
        i = Interval(-5, -2)
        self.assertEqual(i.start, -5)
        self.assertEqual(i.end, -2)
        self.assertTrue(-4 in i)
        self.assertEqual(hash(i), hash(Interval(-5, -2)))
        self.assertEqual(str(i), "[-5, -2]")
        self.assertEqual(repr(i), "[-5, -2]")

        self.assertFalse(i.overlaps(Interval(1, 2)))
        self.assertTrue(i.overlaps(Interval(-6, -4)))
        self.assertTrue(i.overlaps(Interval(-6, -5)))
        self.assertTrue(i.overlaps(Interval(-3, -1)))
        self.assertEqual(i.intersection(Interval(-6, -4)), Interval(-5, -4))
        self.assertEqual(i.intersection(Interval(-3, -1)), Interval(-3, -2))
        self.assertIsNone(i.intersection(Interval(1, 2)))
        self.assertEqual(i.union(Interval(-6, -4)), Interval(-6, -2))
        self.assertEqual(i.union(Interval(-3, -1)), Interval(-5, -1))

    def test_interval_negative_values2(self):
        # Test when start and end values are both negative
        i = Interval(-5, -2)
        self.assertEqual(i.start, -5)
        self.assertEqual(i.end, -2)
        self.assertTrue(-3 in i)
        self.assertEqual(hash(i), hash(Interval(-5, -2)))
        self.assertEqual(str(i), "[-5, -2]")
        self.assertEqual(repr(i), "[-5, -2]")

        self.assertFalse(i.overlaps(Interval(-1, 0)))
        self.assertFalse(i.overlaps(Interval(1, 2)))
        self.assertTrue(i.overlaps(Interval(-6, -5)))
        self.assertTrue(i.overlaps(Interval(-2, -1)))

        self.assertIsNone(i.intersection(Interval(-1, 0)))
        self.assertEqual(i.intersection(Interval(-6, -4)), Interval(-5, -4))
        self.assertEqual(i.union(Interval(-1, 0)), Interval(-5, 0))
        self.assertEqual(i.union(Interval(-6, -7)), Interval(-6, -2))
