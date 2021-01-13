
from copy import deepcopy
from numpy import where, minimum, maximum


def imputate_with_regression(
		data, regressor, based_on_col, group_by = None,
		cols=None, except_cols=[], inplace=False,
		limit_min = False,
		limit_max = False,
		global_min = None,
		global_max = None
):

	if inplace:
		new_data = data
	else:
		new_data = data.copy()

	if cols is None:
		cols = new_data.columns

	if group_by is None:

		for col in cols:
			if col not in except_cols and col != based_on_col and col and new_data[col].isnull().values.any():
				available_data = data[data[col].isnull()==False]

				this_regressor = deepcopy(regressor)
				try:
					this_regressor.train(X=available_data[[based_on_col]], y=available_data[col], echo=False)
					pred = this_regressor.predict(data=data[[based_on_col]], echo=False)
					if limit_min:
						if global_min is not None:
							the_min = global_min
						else:
							the_min = available_data[col].min()
						pred = maximum(pred, the_min)

					if limit_max:
						if global_max is not None:
							the_max = global_max
						else:
							the_max = available_data[col].max()
						pred = minimum(pred, the_max)

					new_data[col] = where(data[col].isnull(), pred, data[col])
				except:
					pass

	else:
		return data.groupby(group_by).apply(
			lambda x: imputate_with_regression(
				data=x, based_on_col=based_on_col, cols=cols, except_cols=except_cols,
				inplace=False, regressor=regressor, limit_min=limit_min, limit_max=limit_max,
				global_min=global_min, global_max=global_max
			)
		).reset_index(drop=True)

	return new_data