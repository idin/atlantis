from ..evaluation import evaluate_regression, evaluate_classification


def train_and_test(
		estimator, evaluation_type, training_data, test_data, x_columns, y_column, sort_columns,
		ignore_missing_y=False, ignore_missing_x=False, replace_missing_x='mean'
):
	training_data = training_data[x_columns + [y_column]]

	if ignore_missing_y and ignore_missing_y:
		training_data = training_data.drop_na()
		test_data = test_data.drop_na()

	elif ignore_missing_y:
		training_data = training_data[training_data[y_column].notna()]
		test_data = test_data[training_data[y_column].notna()]

	elif ignore_missing_x:
		training_data = training_data[~training_data[x_columns].isnull().any(axis=1)]
		test_data = test_data[~test_data[x_columns].isnull().any(axis=1)]

	if not ignore_missing_x:
		if replace_missing_x == 'mean':
			training_data[x_columns] = training_data[x_columns].fillna(training_data[x_columns].mean())
			test_data[x_columns] = test_data[x_columns].fillna(test_data[x_columns].mean())

		elif replace_missing_x == 'median':
			training_data[x_columns] = training_data[x_columns].fillna(training_data[x_columns].median())
			test_data[x_columns] = test_data[x_columns].fillna(test_data[x_columns].median())

	estimator.fit(X=training_data[x_columns], y=training_data[y_column])
	actual = test_data[y_column]
	predicted = estimator.predict(test_data[x_columns])
	if evaluation_type.lower().startswith('regress'):
		evaluation = evaluate_regression(actual=actual, predicted=predicted)
	elif evaluation_type.lower().startswith('class'):
		evaluation = evaluate_classification(actual=actual, predicted=predicted)

	evaluation = {
		'training_size': training_data.shape[0],
		'test_size': test_data.shape[0],
		**evaluation
	}

	if sort_columns is not None:
		sort_values = {}
		for sort_column in sort_columns:
			sort_values[f'training_from_{sort_column}'] = training_data[sort_column].min()
			sort_values[f'training_to_{sort_column}'] = training_data[sort_column].max()
			sort_values[f'test_from_{sort_column}'] = test_data[sort_column].min()
			sort_values[f'test_to_{sort_column}'] = test_data[sort_column].max()
		evaluation = {**sort_values, **evaluation}
	return evaluation
