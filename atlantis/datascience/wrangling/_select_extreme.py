def _find_max_row(data, col):
	return data[data.index == data[col].idxmax()]

def _find_min_row(data, col):
	return data[data.index == data[col].idxmin()]

def select_extreme(data, col, group_by=None, extreme_type = 'max'):
	if extreme_type=='max':
		if group_by is None:
			return _find_max_row(data=data, col=col)
		else:
			return data.groupby(by = group_by).apply(lambda x: _find_max_row(data=x, col=col))
	else:
		if group_by is None:
			return _find_min_row(data=data, col=col)
		return data.groupby(by = group_by).apply(lambda x: _find_min_row(data=x, col=col))

def select_max(data, max_col, group_by=None):
	return select_extreme(data = data, col = max_col, group_by = group_by, extreme_type = 'max')

def select_min(data, min_col, group_by=None):
	return select_extreme(data = data, col = min_col, group_by = group_by, extreme_type = 'min')