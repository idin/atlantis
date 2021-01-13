import pandas as pd

def fast_merge(left, right, left_on, right_on, how='inner'):
	"""
	:type left: pd.DataFrame
	:type right: pd.DataFrame
	:type left_on: list[str]
	:type right_on: list[str]
	:type how: str
	:rtype: pd.DataFrame
	"""

	result = left.set_index(keys=left_on, drop=True).join(
		other=right.set_index(keys=right_on, drop=True),
		how=how, lsuffix='_x', rsuffix='_y'
	)
	result[left_on] = result.index
	result[right_on] = result.index
	result.reset_index(drop=True, inplace=True)

	return result