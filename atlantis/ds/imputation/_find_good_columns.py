from ..wrangling import count_rows_without_missing
from ...collections import get_powerset


def find_columns_that_fully_cover_missing(data, column):
	"""
	if a column has missing values,
	find other columns such that these "helper columns" always have values
	for rows with missing values of the column in question
	"""
	rows_with_na = data[data[column].isna()]
	return [name for name, is_na in dict(rows_with_na.isna().any()).items() if not is_na]


def count_non_na_per_column(data):
	return dict(data.notna().sum())


def get_value_coverage(data, column):
	"""
	for a certain column with missing value,
	find how many values does every other column have for rows with missing values of the column
	"""
	non_na = data[data[column].notna()].drop(columns=column)
	return count_non_na_per_column(non_na)


def get_missing_coverage_per_column_subset(data, column):
	"""
	if a column has missing values, consider all subsets of other columns and see how much
	they cover
	"""
	viable_columns = find_columns_that_fully_cover_missing(data=data, column=column)
	non_empty_combinations = [l for l in get_powerset(iterable=viable_columns) if len(l) > 0]
	results = []
	for combination in non_empty_combinations:
		sub_data = data[combination]
		coverage = count_rows_without_missing(data=sub_data)
		column_ratio = len(combination)/(data.shape[1] - 1)
		row_ratio = coverage/data.shape[0]
		result = {
			'combination': combination,
			'row_coverage': coverage, 'row_ratio': row_ratio,
			'column_ratio': column_ratio,
			'row_squared_column_score': row_ratio ** 2 * column_ratio
		}
		results.append(result)
	return results


def find_best_coverage_combination(data, column, method='row_squared_column_score'):
	combinations = get_missing_coverage_per_column_subset(data=data, column=column)
	return sorted(combinations, key=lambda x: -x[method])[0]
