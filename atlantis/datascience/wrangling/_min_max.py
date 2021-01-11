import pandas as pd
_min = min
_max = max


def min(data, axis=1, skip_na=True):
	"""

	:type data: pd.DataFrame
	:param axis:
	:return:
	"""
	if not skip_na:
		return data.min(axis=axis)

	def min_except_na(l):
		l2 = [x for x in l if x is not None]
		if len(l2)==0:
			return None
		else:
			return _min(l2)

	return data.apply(min_except_na, axis=axis)


def max(data, axis=1, skip_na=True):
	"""

	:type data: pd.DataFrame
	:param axis:
	:return:
	"""
	if not skip_na:
		return data.max(axis=axis)

	def max_except_na(l):
		l2 = [x for x in l if x is not None]
		if len(l2)==0:
			return None
		else:
			return _max(l2)

	return data.apply(max_except_na, axis=axis)