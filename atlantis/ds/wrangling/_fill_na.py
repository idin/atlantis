def fill_na(data, replace_with=0, cols=None, except_cols=None, na_col_suffix='_na', inplace=False, na_col_as_int=False):
	except_cols = except_cols or []

	if inplace:
		new_data = data
	else:
		new_data = data.copy()

	if cols is None:
		cols = new_data.columns

	for col in cols:
		if col not in except_cols and new_data[col].isnull().values.any():
			na_values = new_data[col].isnull().values
			if na_col_as_int:
				na_values = na_values.astype(int)  # to turn boolean to int
			new_data[str(col) + na_col_suffix] = na_values
			new_data[col].fillna(value=replace_with, inplace=True)

	return new_data
