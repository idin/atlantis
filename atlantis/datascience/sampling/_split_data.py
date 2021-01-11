from pandas import DataFrame
from math import ceil


def split_data(data, split_size=None, num_splits=None):
	"""
	:type data: DataFrame
	:type split_size: int
	:type num_splits: int
	:rtype: generator
	"""
	if split_size is None and num_splits is None:
		raise ValueError('either size or num_splits should be provided')
	if split_size is None:
		split_size = len(data) / num_splits
	if num_splits is None:
		num_splits = ceil(len(data) / split_size)

	for i in range(num_splits):
		start = round(i * split_size)
		end = min(round((i + 1) * split_size), len(data))
		this_data = data.iloc[start:end]
		yield this_data
