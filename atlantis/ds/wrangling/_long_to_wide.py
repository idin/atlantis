from ._standardize_columns import standardize_columns as sc


def long_to_wide(data, index_by, widen_cols, value_cols=None, standardize_cols=True):
	"""
	:type data: DataFrame
	:type index_by: list[str]
	:type widen_cols: list[str]
	:type value_cols: list[str]
	:type standardize_cols: bool
	:rtype: DataFrame
	"""
	result = data.pivot_table(
		aggfunc=lambda x: x,
		index=index_by,
		columns=widen_cols,
		values=value_cols
	).reset_index()

	if standardize_cols:
		sc(data=result, inplace=True)
	return result


tall_to_wide = long_to_wide
