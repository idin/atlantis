import pandas as pd
from txt import convert_camel_to_snake
from .wrangling import standardize_columns as standardize


def read_excel(path, standardize_columns=True):
	excel = pd.ExcelFile(path)
	result = {
		convert_camel_to_snake(name): pd.read_excel(path, sheet_name=name)
		for name in excel.sheet_names
	}

	if standardize_columns:
		result = {key: standardize(value) for key, value in result.items()}

	return result
