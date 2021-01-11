from pandas import DataFrame


def balance_sample(data, columns, n=None, fraction=None, random_state=None):
	"""
	this method takes a sample of the data by reducing the imbalance between the values of columns
	:type data: DataFrame
	:type columns: str or list[str]
	:type n: int or NoneType
	:type fraction: float or NoneType
	:type random_state: NoneType or float or int
	"""

	if n is None:
		if fraction > 1:
			raise ValueError('fraction cannot be larger than 1')
		n = round(fraction * data.shape[0])
	if n > data.shape[0]:
		raise ValueError('n cannot be larger than the number of rows')

	n = min(n, data.shape[0])

	def add_index(x):
		x = x.sample(frac=1, random_state=random_state)
		x['__balance_sample_index__'] = range(x.shape[0])
		return x

	new_data = data.groupby(columns).apply(add_index)
	new_data.index = new_data.index.droplevel(0)
	new_data = new_data.sort_values('__balance_sample_index__').head(n)

	return new_data.drop(columns='__balance_sample_index__')
