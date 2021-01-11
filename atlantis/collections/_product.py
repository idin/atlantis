def _get_product(l1, l2):
	"""
	crosses two lists and returns tuples of form (x, y) for all x in l1 and all y in l2
	:rtype: list[tuple]
	"""
	return [(x, y) for x in l1 for y in l2]


def get_product(*args):
	"""
	crosses n lists and returns tuples of length n where n is the number of lists where
	each tuple is (x1, x2, ...xn) and xi is any element of list i.
	:rtype: list[tuple]
	"""
	if len(args) == 2:
		return _get_product(args[0], args[1])
	elif len(args) > 2:
		result = get_product(_get_product(args[0], args[1]), *args[2:])
		return [(x[0][0], x[0][1], *x[1:]) for x in result]
	elif len(args) == 1 and isinstance(args[0], (list, tuple)):
		return get_product(*args[0])
