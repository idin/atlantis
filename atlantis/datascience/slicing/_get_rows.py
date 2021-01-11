from ...collections import OrderedSet


def get_rows(data):
	try:
		return data.rows
	except AttributeError:
		return OrderedSet(range(data.shape[0]))


def get_columns(data):
	columns = data.columns
	if isinstance(columns, OrderedSet):
		return columns
	else:
		return OrderedSet(columns)
