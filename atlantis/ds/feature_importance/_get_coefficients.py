def get_coefficients(model, columns, raise_error=True, **kwargs):
	try:
		coef = model.coef_
		if isinstance(coef, dict):
			return coef
		try:
			coefficients = list(coef.flatten())
		except:
			coefficients = list(coef)
	except (AttributeError, KeyError):
		return None

	try:
		intercept = model.intercept_
	except AttributeError:
		intercept = None


	if len(columns) != len(coefficients):
		if raise_error:
			raise RuntimeError(f'number of columns: {len(columns)}, number of coefficients: {len(coefficients)}')
		else:
			return None
	else:
		coefficient_dictionary = {column: coefficient for coefficient, column in zip(coefficients, columns)}

		for key, value in kwargs.items():
			coefficient_dictionary[key] = value

		if intercept is not None and 'intercept' not in coefficient_dictionary:
			coefficient_dictionary['intercept'] = intercept

		return coefficient_dictionary
