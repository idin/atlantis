from itertools import chain, combinations


def get_powerset(iterable):
	"powerset([1,2,3]) --> [[], [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]"
	s = list(iterable)
	return [list(l) for l in chain.from_iterable(combinations(s, r) for r in range(len(s)+1))]


get_set_of_subsets = get_powerset
