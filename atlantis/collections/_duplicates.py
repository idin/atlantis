from collections import Counter
from ._make_hashable import make_hashable
from .Counter import Counter


def has_duplicates(l):
	"""
	returns True if list l has duplicates
	:type l: list
	:rtype: bool
	"""
	if len(l) == 0:
		return False
	else:
		return Counter(l).most_common()[0][1] > 1


def get_duplicates(l):
	"""
	returns duplicate values in list l
	:type l: list
	:rtype: list
	"""
	return l.__class__([k for k, v in Counter(l).items() if v > 1])


def remove_duplicates(l):
	"""
	removes the duplicates from a list l
	:type l: list
	:rtype: list
	"""
	used = set()
	unique_list = l.__class__([x for x in l if make_hashable(x) not in used and (used.add(make_hashable(x)) or True)])
	return unique_list
