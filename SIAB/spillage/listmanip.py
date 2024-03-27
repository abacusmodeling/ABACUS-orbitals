def flatten(x):
    '''
    Flattens a nested list.

    '''
    def _recursive_flatten(x):
        return [elem for i in x for elem in _recursive_flatten(i)] if isinstance(x, list) else [x]

    assert isinstance(x, list)
    return _recursive_flatten(x)


def nest(x, pattern):
    '''
    Nests a list according to a nesting pattern.
    
    Parameters
    ----------
        x : list
            A list to be nested.
        pattern : (nested) list
            A nested list of non-negative int that specifies the nesting pattern.
    
    Examples
    --------
    >>> x = [1,2,3,4,5]
    >>> pattern = [[2], [2], [1]]
    >>> print(nest(x, pattern))
    [[1, 2], [3, 4], [5]]
    >>> pattern = [[2], 2, [1], [0]]
    >>> print(nest(x, pattern))
    [[1, 2], 3, 4, [5], []]
    
    Notes
    -----
    Sublists must be specified in the pattern by enclosing their sizes in lists.
    For example, to nest [1,2,3,4,5] into [[1,2],[3,4,5]], one needs a pattern of [[2],[3]];
    [2,3] would not do anything. Empty lists must also be specified explicitly as [0].
    
    '''
    def _nestgen(x, pattern):
        idx = 0
        for i in pattern:
            if isinstance(i, list):
                stride = sum(flatten(i))
                yield list(_nestgen(x[idx:idx+stride], i))
            else:
                stride = i
                yield from x[idx:idx+stride]
            idx += stride

    assert isinstance(x, list) and isinstance(pattern, list)
    assert len(x) == sum(flatten(pattern))
    return list(_nestgen(x, pattern))


def nestpat(x):
    '''
    Finds the nesting pattern of a list.
    
    The nesting pattern is a (nested) list of non-negative integers.
    For a plain (i.e., not nested) list of length n, the pattern is [n].
    For a nested list like [[1,2], [[]], 3, 4, [5]], the pattern is [[2], [[0]], 2, [1]].
    
    '''
    def _patgen(x):
        if len(x) == 0:
            yield 0

        streak_count = 0 # number of consecutive non-list elements
        for xi in x:
            if isinstance(xi, list):
                if streak_count > 0:
                    yield streak_count
                    streak_count = 0
                yield list(_patgen(xi))
            else:
                streak_count += 1

        if streak_count > 0: # trailing non-list elements
            yield streak_count

    assert isinstance(x, list)
    return list(_patgen(x))


def merge(l1, l2, depth):
    '''
    Merges two (nested) lists at a specified depth.

    A depth-0 merge concatenates two lists.
    A depth-n merge applies a depth-(n-1) merge to two lists element-wise.
    If the two lists are of different lengths, the longer part is simply
    appended.

    '''
    def _mergegen(l1, l2, depth):
        if depth == 0:
            yield from l1 + l2
        else:
            yield from (list(_mergegen(i, j, depth-1)) for i, j in zip(l1, l2))
            if len(l1) != len(l2):
                l_long, l_short = (l1, l2) if len(l1) > len(l2) else (l2, l1)
                yield from l_long[len(l_short):]
    
    assert isinstance(l1, list) and isinstance(l2, list)
    assert isinstance(depth, int) and depth >= 0
    # maybe some depth checking?
    return list(_mergegen(l1, l2, depth))


############################################################
#                       Test
############################################################
import unittest
class _TestListManip(unittest.TestCase):

    def test_flatten(self):
        x = [[[]]]
        self.assertEqual(flatten(x), [])
    
        x = [[], [], []]
        self.assertEqual(flatten(x), [])
    
        x = [[[], 1]]
        self.assertEqual(flatten(x), [1])
    
        x = [[1, [2, 3], [], 4], [[[5]]], [[]]]
        self.assertEqual(flatten(x), [1, 2, 3, 4, 5])
    
    
    def test_nest(self):
        x = []
        pattern = [[[0]]]
        self.assertEqual(nest(x, pattern), [[[]]])
    
        x = [7]
        pattern = [[0], [0], [[1]]]
        self.assertEqual(nest(x, pattern), [[], [], [[7]]])
    
        x = [0, 1, 2, 3, 4]
        pattern = [[3], 2, [0]]
        self.assertEqual(nest(x, pattern), [[0, 1, 2], 3, 4, []])
    
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        pattern = [[1, [2], [[1]]], [[2], 1], [1], 2]
        self.assertEqual(nest(x, pattern), [[0, [1, 2], [[3]]], [[4, 5], 6], [7], 8, 9])
    
    
    def test_nestpat(self):
        x = [1, 2, 3]
        self.assertEqual(nestpat(x), [3])
    
        x = [[1, 2], 3, [[4, 5], 6], 7]
        self.assertEqual(nestpat(x), [[2], 1, [[2], 1], 1])
    
        x = [[1, 2], 3, 4, [[]], [5], []]
        self.assertEqual(nestpat(x), [[2], 2, [[0]], [1], [0]])
    
    
    def test_merge(self):
        l1 = []
        l2 = []
        self.assertEqual(merge(l1, l2, 0), [])
    
        l1 = [7]
        l2 = []
        self.assertEqual(merge(l1, l2, 0), [7])
    
        l1 = [0]
        l2 = [1, 2]
        self.assertEqual(merge(l1, l2, 0), [0, 1, 2])
    
        l1 = [[0, 1], [2, 3]]
        l2 = [[], [4, 5], [[6, 7]]]
        self.assertEqual(merge(l1, l2, 0), [[0, 1], [2, 3], [], [4, 5], [[6, 7]]])
        self.assertEqual(merge(l1, l2, 1), [[0, 1], [2, 3, 4, 5], [[6, 7]]])
    
        l1 = [[[0], [1]], [[2, 3]]]
        l2 = [[[4, 5]], [[6], [7]]]
        self.assertEqual(merge(l1, l2, 1), [[[0], [1], [4, 5]], [[2, 3], [6], [7]]])
        self.assertEqual(merge(l1, l2, 2), [[[0, 4, 5], [1]], [[2, 3, 6], [7]]])


if __name__ == '__main__':
    unittest.main()

