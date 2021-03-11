from atlantis.ds.parallel_computing._Task import Task
import traceback
from pandas import DataFrame
import shap
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import matplotlib.pyplot as plt
from ...feature_importance import get_feature_importances
from .._get_data_from_namespace import get_obj_from_namespace, get_data_from_namespace


class LearningTask(Task):
	def __init__(
			self, project_name, estimator_class, estimator_name, estimator_id, estimator_arguments,
			training_test_slice_id, y_column, x_columns, evaluation_function,
	):

		super().__init__(project_name=project_name, task_id=None)
		if not isinstance(estimator_id, (str, int)):
			raise TypeError('estimator_id should be an int or str')

		if not isinstance(project_name, (str, int)):
			raise TypeError('project_name should be either an int or a str')

		if not isinstance(estimator_class, type):
			raise TypeError('estimator_class should be a type')

		if not isinstance(estimator_arguments, dict):
			raise TypeError('kwargs should be an int')

		if not isinstance(y_column, str):
			raise TypeError('y_column should be a str')

		self._estimator_id = estimator_id
		self._estimator_class = estimator_class
		self._estimator_name = estimator_name
		self._estimator_arguments = estimator_arguments

		self._training_test_id = training_test_slice_id
		self._y_column = y_column
		self._x_columns = x_columns
		self._evaluation_function = evaluation_function

		self._evaluation = None
		self._id = self.project_name, self.estimator_name, self.estimator_id, self.training_test_id, self.y_column
		self._predictions = None
		self._trained_estimator = None
		self._feature_importances = None
		self._shap_values = None
		self._training_x = None

	@property
	def training_test_id(self):
		return self._training_test_id

	@property
	def time_estimate_id(self):
		return self.estimator_name

	@property
	def status(self):
		return self._status

	def __hash__(self):
		return hash(self.id)

	@property
	def estimator_id(self):
		return self._estimator_id

	@property
	def estimator_class(self):
		return self._estimator_class

	@property
	def estimator_name(self):
		return self._estimator_name

	@property
	def estimator_arguments(self):
		return self._estimator_arguments

	@property
	def y_column(self):
		return self._y_column

	@property
	def x_columns(self):
		return self._x_columns

	@property
	def evaluation(self):
		return self._evaluation

	def evaluate(self, actual, predicted):
		self._evaluation = self._evaluation_function(actual=actual, predicted=predicted)

	@property
	def record(self):
		"""
		:rtype: dict
		"""
		evaluation = self.evaluation or {}

		return {
			'project_name': self.project_name,
			'estimator_name': self.estimator_name,
			'estimator_id': self.estimator_id,
			'training_test_id': self._training_test_id,
			'worker_id': self._worker_id,
			'status': self._status,
			'starting_time': self.starting_time,
			'ending_time': self.ending_time,
			'elapsed_ms': self.get_elapsed(unit='ms'),
			**evaluation
		}

	@property
	def predictions(self):
		"""
		:rtype: DataFrame
		"""
		if self._predictions is None:
			raise RuntimeError('predictions not available. you should run do() with return_predictions=True')
		return self._predictions

	@property
	def trained_estimator(self):
		"""
		:rtype: LinearRegression
		"""
		if self._trained_estimator is None:
			raise RuntimeError('trained estimator not available. you should run do() with return_predictions=True')
		return self._trained_estimator

	@property
	def feature_importances(self):
		"""
		:rtype: dict
		"""
		return self._feature_importances

	def do(self, namespace, worker_id, return_predictions=False):
		"""
		:type namespace: Namespace
		:type worker_id: int or str
		"""
		try:
			self.start()

			estimator = self.estimator_class(**self.estimator_arguments)
			training_test_slice = get_obj_from_namespace(
				namespace=namespace,
				obj_type='tts', obj_id=self.training_test_id
			)
			data = get_data_from_namespace(namespace=namespace, data_id=training_test_slice.data_id)

			training_data = training_test_slice.get_training_data(data=data)
			training_data = training_data[training_data[self.y_column].notna()]

			test_data = training_test_slice.get_test_data(data=data)

			training_x = training_data[self.x_columns]
			training_y = training_data[self.y_column]
			estimator.fit(X=training_x, y=training_y)

			test_x = test_data[self.x_columns]
			actual_all = test_data[self.y_column]
			actual_evaluation = test_data[test_data[self.y_column].notna()][self.y_column]
			predicted_all = estimator.predict(test_x)
			predicted_evaluation = predicted_all[test_data[self.y_column].notna()]

			if return_predictions:
				result = test_x
				result['actual'] = actual_all
				result['predicted'] = predicted_all
				self._trained_estimator = estimator
				self._predictions = result
				self._feature_importances = get_feature_importances(model=estimator, columns=self.x_columns)
				if isinstance(
					self.trained_estimator,
					(
						XGBRegressor, XGBClassifier,
						RandomForestClassifier, RandomForestRegressor,
						DecisionTreeRegressor, DecisionTreeClassifier
					)
				):
					explainer = shap.TreeExplainer(self.trained_estimator)
					self._shap_values = explainer.shap_values(training_x)
					self._training_x = training_x

			self.evaluate(actual=actual_evaluation, predicted=predicted_evaluation)
			if self._evaluation is None:
				raise RuntimeError('evaluation is None')

			self.end(worker_id=worker_id)
		except Exception as error:
			self.add_error(error=error, trace=traceback.format_exc())

	def plot_shap(self, plot_type='bar', show=True, path=None, width=16, height=12):
		if show:
			shap.summary_plot(self._shap_values, self._training_x, show=show, plot_type=plot_type)
		else:
			figure = plt.gcf()
			if width is not None and height is not None:
				figure.set_size_inches(width, height)
			shap.summary_plot(self._shap_values, self._training_x, show=False, plot_type=plot_type)
			plt.savefig(path, dpi=300, bbox_inches='tight')
