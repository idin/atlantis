from pandas import DataFrame

def join_and_preserve_types(left, right, priority='left', **kwargs):
	"""
	:type left: DataFrame
	:type right: DataFrame
	:type priority: str
	:type kwargs: dict
	:rtype: DataFrame
	"""

	if priority=='left':
		orig = left.dtypes.to_dict()
		orig.update(right.dtypes.to_dict())
	else:
		orig = right.dtypes.to_dict()
		orig.update(left.dtypes.to_dict())

	joined = left.merge(right=right, **kwargs)

	def _fix_type(x):
		try:
			return x.astype(orig[x.name])
		except:
			return x

	return joined.apply(lambda x: _fix_type(x))