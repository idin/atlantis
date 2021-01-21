import numpy as np
from pandas import DataFrame


def count_rows_with_missing(data):
	"""
	:type data: DataFrame
	:rtype: int
	"""
	return np.count_nonzero(np.count_nonzero(data.isnull(), axis=1))


def count_rows_without_missing(data):
	"""
	:type data: DataFrame
	:rtype: int
	"""
	return data.shape[0] - count_rows_with_missing(data=data)
