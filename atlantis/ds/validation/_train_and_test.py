from ..evaluation import evaluate_regression, evaluate_classification


def evaluate(estimator, evaluation_type, training_data, test_data, x_columns, y_column, sort_columns):
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
