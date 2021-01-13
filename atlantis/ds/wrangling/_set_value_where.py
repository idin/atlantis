from pandas import DataFrame, Series

def set_value_where(data, columns, value, where):
	"""
	:type data: DataFrame
	:type columns: str
	:type where: Series
	:rtype: DataFrame
	"""

	data.set_value(index=data[where].index, col=columns, value=value)
	return data
