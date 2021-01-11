import pandas as pd

def find_duplicated_columns(data):
	"""
	:type data: pd.DataFrame
	:rtype: list[str]
	"""
	columns = data.columns.to_series()
	return list(columns[columns.duplicated()])