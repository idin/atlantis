import pandas as pd
from ...text import convert_camel_to_snake
from ._standardize_columns import standardize_columns as standardize


def read_excel(
		path, standardize_columns=True, headers=0, column_names=None, index_columns=None,
		find_header_by_columns=None, ignore_sheets_with_no_header=False
):
	"""
	:param str 						path: 			the path to the excel file
	:param bool 		standardize_columns: 		if True the columns will be converted to snake_case
	:param int or dict[str, int] 	headers: 		header index, if dictionary then each sheet will use the respective
													one based on the key and the sheet name
	:param list or dict[str, list] 	column_names: 	explicit column_names, also dictionary is used to point to each sheet
	:param int or dict[str, int] 	index_columns: 	tells which column should be used as index, can also be dictionary

	:param dict[str, list] 	find_header_by_columns: if the header is not obvious, this allows the function to find it
	:param bool		ignore_sheets_with_no_headers: 	relevant when find_header_by_columns, if True,
													when header is not found, sheet is ignored
	"""
	excel = pd.ExcelFile(path)
	result = {}
	for sheet_name in excel.sheet_names:
		snake_name = convert_camel_to_snake(sheet_name)

		if isinstance(headers, dict):
			if sheet_name in headers:
				_header = headers[sheet_name]
			elif snake_name in headers:
				_header = headers[snake_name]
			else:
				raise KeyError(f'neither {sheet_name} nor {snake_name} in header dictionary!')
		else:
			_header = headers

		if isinstance(column_names, dict):
			if sheet_name in column_names:
				_column_names = column_names[sheet_name]
			elif snake_name in column_names:
				_column_names = column_names[snake_name]
			else:
				raise KeyError(f'neither {sheet_name} nor {snake_name} in column_names dictionary!')
		else:
			_column_names = column_names

		if isinstance(index_columns, dict):
			if sheet_name in index_columns:
				_index_column = index_columns[sheet_name]
			elif snake_name in index_columns:
				_index_column = index_columns[snake_name]
			else:
				raise KeyError(f'neither {sheet_name} nor {snake_name} in index_columns dictionary!')
		else:
			_index_column = index_columns

		if isinstance(find_header_by_columns, dict):
			if sheet_name in find_header_by_columns:
				_find_header_by_columns = find_header_by_columns[sheet_name]
			elif snake_name in find_header_by_columns:
				_find_header_by_columns = find_header_by_columns[snake_name]
			else:
				raise KeyError(f'neither {sheet_name} nor {snake_name} in find_header_by_columns dictionary!')
		else:
			_find_header_by_columns = find_header_by_columns

		if _find_header_by_columns is None:
			result[snake_name] = pd.read_excel(
				path, sheet_name=sheet_name, header=_header, names=_column_names, index_col=_index_column
			)
		else:
			temp = pd.read_excel(
				path, sheet_name=sheet_name, header=None, names=_column_names, index_col=_index_column
			)
			if not isinstance(_find_header_by_columns, list):
				_find_header_by_columns = [_find_header_by_columns]
			for i, row in temp.iterrows():
				if all(
						convert_camel_to_snake(x, ignore_errors=True) == convert_camel_to_snake(y, ignore_errors=True)
						for x, y in zip(_find_header_by_columns, row)
				):
					result[snake_name] = pd.read_excel(
						path, sheet_name=sheet_name, header=i, names=_column_names, index_col=_index_column
					)
					break
			else:
				if ignore_sheets_with_no_header:
					continue
				else:
					raise KeyError(f'could not find the header using columns: {_find_header_by_columns}')

	if standardize_columns:
		result = {key: standardize(value) for key, value in result.items()}

	return result
