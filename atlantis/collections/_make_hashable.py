def is_hashable(obj):
	try:
		hash(obj)
		return True
	except TypeError:
		return False


def make_hashable(obj):
	try:
		return make_hashable(obj.__hashkey__())
	except AttributeError:
		if isinstance(obj, (tuple, list)):
			return tuple((make_hashable(e) for e in obj))

		if isinstance(obj, dict):
			return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))

		if isinstance(obj, (set, frozenset)):
			return tuple(sorted(make_hashable(e) for e in obj))

		return obj
