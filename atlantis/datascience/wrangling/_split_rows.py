from pandas import concat, Series

from ...text import split_and_strip

# splits one column of a dataframe into multiple rows
# for example: the following dataframe
## number	label
## 1,2		something

# becomes:
## number


def split_rows(data, by_column, sep=',', split_column_name=None, keep_original=False):
	x = data[[by_column]]
	if split_column_name is None:
		split_column_name = by_column + '_split'
	split_data = concat([Series(row[by_column], split_and_strip(row[by_column], sep=sep)) for _, row in x.iterrows()]).reset_index()
	split_data.columns = [split_column_name, by_column]

	# if there is a column called by_column + _split in the original data there would be trouble
	result = split_data.merge(right=data, on=by_column, how='outer')

	if not keep_original:
		result.drop(axis=1, labels=by_column, inplace=True)
		result.rename(columns={split_column_name: by_column}, inplace=True)
	return result
