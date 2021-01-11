def select_columns(data, include = None, exclude = None):
	if include is None:
		include = list(data.columns)

	if exclude is None:
		exclude = []

	columns = [column for column in include if column not in exclude]
	return data[columns]
