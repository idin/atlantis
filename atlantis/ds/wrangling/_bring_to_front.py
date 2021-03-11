from pandas import DataFrame


def move_columns(data, to, columns=None):
	"""
	brings the columns to the front of dataframe
	:type data: DataFrame
	:param str to: 'front' brings to front, 'back' sends to back
	:type columns: list[str] or NoneType
	:rtype: DataFrame or NoneType
	"""
	columns = columns or []
	if isinstance(columns, str):
		columns = [columns]
	other_columns = [column for column in data.columns if column not in columns]
	if to == 'front':
		return data[columns + other_columns]
	elif to == 'back':
		return data[other_columns + columns]


def bring_to_front(data, columns=None):
	"""
	brings the columns to the front of dataframe
	:type data: DataFrame
	:type columns: list[str] or NoneType
	:rtype: DataFrame or NoneType
	"""
	return move_columns(data=data, to='front', columns=columns)


def send_to_back(data, columns=None):
	"""
	brings the columns to the front of dataframe
	:type data: DataFrame
	:type columns: list[str] or NoneType
	:rtype: DataFrame or NoneType
	"""
	return move_columns(data=data, to='back', columns=columns)
