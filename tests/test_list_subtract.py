import unittest


from beat_the_streak.list_subtract import subtract

class TestListSubtract(unittest.TestCase):
    def test_returns_list_with_subtracted_elements(self):
        new_list = subtract(['a', 'b', 'c'], ['b'])
        assert new_list == ['a', 'c']

