from ...collections import has_duplicates

def columns_are_unique(data):
	return not has_duplicates(list(data.columns))
