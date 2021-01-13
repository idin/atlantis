from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import numpy as np
from warnings import warn, catch_warnings, simplefilter


def rmse(actual, predicted):
	actual = np.array(actual)
	predicted = np.array(predicted)
	return sqrt(mean_squared_error(actual, predicted))


root_mean_squared_error = rmse


def mape(actual, predicted, ignore_warnings=True):
	actual = np.array(actual)
	predicted = np.array(predicted)
	if ignore_warnings:
		with catch_warnings():
			simplefilter('ignore')
			result = np.mean(np.abs((actual - predicted) / ((actual + predicted)/2))) * 100
	else:
		result = np.mean(np.abs((actual - predicted) / ((actual + predicted) / 2))) * 100
	return result


# symmetric mean absolute percentage error
def smape(actual, predicted, ignore_warnings=True):
	actual = np.array(actual)
	predicted = np.array(predicted)
	if ignore_warnings:
		with catch_warnings():
			simplefilter('ignore')

			result = np.mean(np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) / 2)) * 100
	else:
		result = np.mean(np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) / 2)) * 100
	return result


mean_absolute_percentage_error = mape
symmetric_mean_absolute_percentage_error = smape
mae = mean_absolute_error


def nmae(actual, predicted, ignore_warnings=True):
	actual = np.array(actual)
	predicted = np.array(predicted)
	if ignore_warnings:
		with catch_warnings():
			simplefilter('ignore')
			result = mae(actual, predicted)/(actual.max() - actual.min())
	else:
		result = mae(actual, predicted) / (actual.max() - actual.min())
	return result


normalized_mean_absolute_error = nmae


def evaluate_regression(actual, predicted, raise_error=True, ignore_warnings=True):
	actual = np.array(actual)
	predicted = np.array(predicted)

	# remove nas
	predicted = predicted[np.logical_not(np.isnan(actual))]
	actual = actual[np.logical_not(np.isnan(actual))]

	_mae = _rmse = _mape = _nmae = _smape = None
	try:
		_mae = mae(actual, predicted)
		if ignore_warnings:
			with catch_warnings():
				simplefilter('ignore')
				_nmae = _mae / (actual.max() - actual.min())
		else:
			_nmae = _mae / (actual.max() - actual.min())
		_rmse = rmse(actual=actual, predicted=predicted)
		_mape = mape(actual=actual, predicted=predicted, ignore_warnings=ignore_warnings)
		_smape = smape(actual=actual, predicted=predicted, ignore_warnings=ignore_warnings)
	except Exception as e:
		if raise_error:
			raise e
		else:
			message = f'ignoring error in evaluate_regression: {e}'
			print(message)
			warn(message)
	return {'nmae': _nmae, 'rmse': _rmse, 'mae': _mae, 'mape': _mape, 'smape': _smape}
