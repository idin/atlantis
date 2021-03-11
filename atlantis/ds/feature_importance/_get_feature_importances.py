def get_feature_importances(model, columns, raise_error=True, **kwargs):
	try:
		importances = list(model.feature_importances_)
	except AttributeError:
		importances = None

	if importances is None:
		return None

	else:
		if len(columns) != len(importances):
			if raise_error:
				raise RuntimeError(f'number of columns: {len(columns)}, number of features: {len(importances)}')
			else:
				return None
		else:
			importance_dictionary = {column: importance for importance, column in zip(importances, columns)}

			for key, value in kwargs.items():
				importance_dictionary[key] = value

			sorted_dictionary = dict(sorted(importance_dictionary.items(), key=lambda item: -abs(item[1])))

			return sorted_dictionary
