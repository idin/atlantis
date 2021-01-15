from pandas import DataFrame
from pandas.api.types import is_numeric_dtype


def drop_single_value_columns(data, ignore=None, echo=True):
	"""
	:param 	data: a dataframe to be cleaned
	:type 	data: DataFrame
	:type 	echo: bool
	:rtype: DataFrame
	"""
	ignore = ignore or []
	data = data.copy()
	for column in data.columns:
		if column in ignore:
			continue
		uniques = data[column].unique()
		if len(uniques) == 1:
			if echo:
				print(f'dropping single value column: "{column}" with value "{uniques[0]}"')
			data.drop(column, inplace=True, axis=1)
	return data


def drop_columns_with_too_many_nas(data, na_ratio, ignore=None, echo=True):
	"""
	:param 	data: a dataframe to be cleaned
	:type 	data: DataFrame

	:param 	na_ratio: if the ratio of na's in a column (to the number of rows) is above this ratio, drop it
	:type 	na_ratio: float

	:type 	echo: bool

	:rtype: DataFrame
	"""
	data = data.copy()
	ignore = ignore or []
	num_na_in_columns = data.isnull().apply(sum, axis=0)
	if na_ratio >= 1:
		raise ValueError(f'na_ratio should be smaller than 1')

	for column in data.columns:
		if column in ignore:
			continue
		ratio = num_na_in_columns[column] / data.shape[0]
		if ratio >= na_ratio:
			if echo:
				print(f'dropping column "{column}" with na ratio of {round(ratio, 3)}')
				data.drop(columns=column, inplace=True)

	return data


def get_useless_column(data, column_a, column_b):
	"""
	if between column_a and column_b, all the information exists in column_a and we don't need column_b or vice versa
	:param 	data: the dataframe of question
	:type 	data: DataFrame
	:type 	column_a: str
	:type 	column_b: str
	"""

	column_a_uniques = data[[column_a]].drop_duplicates().shape[0]
	column_b_uniques = data[[column_b]].drop_duplicates().shape[0]

	combined_column_uniques = data[[column_a, column_b]].drop_duplicates().shape[0]

	if combined_column_uniques == column_a_uniques == column_b_uniques:
		# either column_a or column_b is useless
		# we prefer to drop non-numeric columns
		if is_numeric_dtype(data[column_a]):
			return column_b
		elif is_numeric_dtype(data[column_b]):
			return column_a
		else:
			# when all equal, second one gets the cut
			return column_b

	if combined_column_uniques == column_b_uniques:
		# column_a doesn't add any information
		return column_a

	elif combined_column_uniques == column_a_uniques:
		# column_b doesn't add any information
		return column_b

	else:
		return None


def drop_columns_with_no_additional_information(data, ignore=None, echo=True):
	useless_columns = set()
	columns = list(data.columns)

	"""
	example: 
	data = pd.DataFrame({
		'a': [4, 4, 4, 4, 5], 
		'b': [1, 2, 1, 1, 3], 
		'c': ['a', 'b', 'a', 'a', 'd'],
		'd': ['a', 'b', 'a', 'a', 'a']
	})
	all columns but b should be dropped
	"""
	ignore = ignore or []
	for i in range(len(columns) - 1):
		column_a = columns[i]
		if column_a in useless_columns or is_numeric_dtype(data[column_a]):
			continue
		else:
			for j in range(i + 1, len(columns)):
				column_b = columns[j]
				if column_b in useless_columns or is_numeric_dtype(data[column_b]):
					continue
				else:
					useless_column = get_useless_column(data=data, column_a=column_a, column_b=column_b)
					if useless_column is not None and useless_column not in ignore:
						useless_columns.add(useless_column)
						if echo:
							print(f'between "{column_a}" and "{column_b}", "{useless_column}" is useless.')

	return data.drop(columns=useless_columns)


def drop_bad_columns(data, na_ratio, ignore=None, echo=True):
	"""
	:param 	data: a dataframe to be cleaned
	:type 	data: DataFrame

	:param 	na_ratio: if the ratio of na's in a column (to the number of rows) is above this ratio, drop it
	:type 	na_ratio: float

	:param 	ignore: list of columns to be totally ignored (get a free pass)
	:type 	ignore: list[str] or set[str]
	:type 	echo: bool

	:rtype: DataFrame
	"""
	data = drop_single_value_columns(data, ignore=ignore, echo=echo)
	data = drop_columns_with_no_additional_information(data=data, ignore=ignore, echo=echo)
	return drop_columns_with_too_many_nas(data=data, na_ratio=na_ratio, ignore=ignore, echo=echo)
