from sklearn.linear_model import LinearRegression
from pandas import DataFrame

from ...time.progress import ProgressBar


class Ensemble:
	def __init__(self, models, weights=None, problem_type='regression', echo=1):
		"""
		:type models: list[LinearRegression]
		:type weights: NoneType of list[float]
		:type problem_type: either 'regression' or 'classification'
		"""
		self._problem_type = problem_type
		self._models = models
		if weights is None:
			self._weights = [1 for model in models]

		else:
			if len(weights) != len(models):
				raise ValueError(f'{len(models)} models need {len(models)} weights!')
			self._weights = weights
		self._fitted = False
		self._echo = echo

	def __repr__(self):
		result = ''
		count = 0
		for weight, model in zip(self._weights, self._models):
			if count >= 3:
				break
			if result != '':
				result += '\n\n'
			result += f'weight: {weight}\n'
			result += repr(model)
			count += 1
		return result

	def fit(self, X, y):
		num_models = len(self._models)
		progress_bar = ProgressBar(total=num_models, echo=self._echo)
		progress = 0
		for model in self._models:
			progress_bar.show(amount=progress, text=f'models fitted: {progress}/{num_models}')
			model.fit(X, y)
			progress += 1
		self._fitted = True
		progress_bar.show(amount=progress, text=f'models fitted: {progress}/{num_models}')

	def predict(self, X):
		if self._fitted is False:
			raise RuntimeError('models are not fitted yet!')

		if self._problem_type.lower()[0] == 'r':
			predictions = [model.predict(X) for model in self._models]

			weighted_sum = None
			for result, weight in zip(predictions, self._weights):
				if weighted_sum is None:
					weighted_sum = result * weight

				else:
					weighted_sum = weighted_sum + (result * weight)

			weighted_average = weighted_sum / sum(self._weights)
			return weighted_average

		else:

			predictions = {f'model_{i}': model.predict(X) for i, model in enumerate(self._models)}
			weights = {f'model_{i}': weight for i, weight in enumerate(self._weights)}
			votes = None

			for model_name, prediction in predictions.items():
				if votes is None:
					votes = DataFrame({model_name: list(prediction)})

				else:
					votes[model_name] = prediction

			def _get_weighted_vote(row):
				result = {}
				for model_name, w in weights.items():
					vote = row[model_name]
					if vote in result:
						result[vote] += w
					else:
						result[vote] = w

				max_score = 0
				winner = None
				for model_name, score in result.items():
					if score > max_score:
						winner = model_name
						max_score = score
				return winner
			return votes.apply(_get_weighted_vote, axis=1)
