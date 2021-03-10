from sklearn.metrics import confusion_matrix
import numpy as np


def evaluate_classification(actual, predicted, labels=None):
	if isinstance(actual, list):
		actual = np.array(actual)
	if isinstance(predicted, list):
		predicted = np.array(predicted)

	# remove nas
	try:
		predicted = predicted[np.logical_not(np.isnull(actual))]
		actual = actual[np.logical_not(np.isnan(actual))]
	except:
		pass

	try:
		if labels is None:
			cm = confusion_matrix(actual, predicted)
		else:
			cm = confusion_matrix(actual, predicted, labels=labels)
	except Exception as e:
		display(actual, predicted)
		raise e

	try:
		true_negative, false_positive, false_negative, true_positive = cm.ravel()
	except:
		print(actual, type(actual), predicted, type(predicted), cm, cm.ravel())
		raise

	positive = false_negative + true_positive
	negative = true_negative + false_positive
	sensitivity = true_positive/positive
	specificity = true_negative/negative
	precision = true_positive/(true_positive + false_positive)
	negative_predictive_value = true_negative/(true_negative + false_negative)
	miss_rate = false_negative/positive
	fall_out = false_positive/negative
	false_discovery_rate = false_positive/(false_positive + true_positive)
	false_omission_rate = false_negative/(false_negative + true_negative)
	threat_score = true_positive/(true_positive + false_negative + false_positive)
	accuracy = (true_positive + true_negative)/(positive + negative)
	f1_score = 2*true_positive/(2*true_positive + false_positive + false_negative)
	try:
		matthews_correlation_coefficient = (true_positive * true_negative - false_positive * false_negative) / np.sqrt(
			(true_positive + false_positive) * (true_positive + false_negative) * (true_negative + false_positive) * (true_negative + false_negative)
		)
	except Exception as e:
		print(e)
		print(
			f'true_positive = {true_positive} ({type(true_positive)})',
			f'true_negative = {true_negative} ({type(true_negative)})',
			f'false_positive = {false_positive} ({type(false_positive)})',
			f'false_negative = {false_negative} ({type(false_negative)})',
			sep='\n'
		)
		raise e
	informedness = sensitivity + specificity - 1
	markedness = precision + negative_predictive_value - 1

	return {
		'accuracy': accuracy,
		'precision': precision,
		'recall': sensitivity,
		'specificity': specificity,

		'negative_predictive_value': negative_predictive_value,
		'miss_rate': miss_rate,
		'fall_out': fall_out,
		'false_discovery_rate': false_discovery_rate,
		'false_omission_rate': false_omission_rate,
		'threat_score': threat_score,
		'f1_score': f1_score,
		'matthews_correlation_coefficient': matthews_correlation_coefficient,
		'informedness': informedness,
		'markedness': markedness,
		'confusion_matrix': cm,
		'true_negative': true_negative,
		'false_positive': false_positive,
		'false_negative': false_negative,
		'true_positive': true_positive
	}
