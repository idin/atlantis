from ._Task import Task


def get_training_data_id(data_id_prefix):
	return f'{data_id_prefix}_training'


def get_test_data_id(data_id_prefix):
	return f'{data_id_prefix}_test'


class LearningTask(Task):
	def __init__(
			self, project_name, estimator_class, estimator_name, estimator_id, estimator_arguments,
			data_id_prefix, y_column, x_columns, evaluation_function,
	):

		super().__init__(project_name=project_name, task_id=None)
		if not isinstance(estimator_id, (str, int)):
			raise TypeError('estimator_id should be an int or str')

		if not isinstance(project_name, (str, int)):
			raise TypeError('project_name should be either an int or a str')

		if not isinstance(data_id_prefix, (int, str)):
			raise TypeError('data_id should be an int or str')

		if not isinstance(estimator_class, type):
			raise TypeError('estimator_class should be a type')

		if not isinstance(estimator_arguments, dict):
			raise TypeError('kwargs should be an int')

		if not isinstance(y_column, str):
			raise TypeError('y_column should be a str')

		self._estimator_id = estimator_id
		self._data_id_prefix = data_id_prefix
		self._estimator_class = estimator_class
		self._estimator_name = estimator_name
		self._estimator_arguments = estimator_arguments
		self._y_column = y_column
		self._x_columns = x_columns
		self._evaluation_function = evaluation_function

		self._evaluation = None
		self._id = self.project_name, self.estimator_name, self.estimator_id, self.data_id_prefix, self.y_column

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
	def data_id_prefix(self):
		return self._data_id_prefix

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
			'data_id': self.data_id_prefix,
			'worker_id': self._worker_id,
			'status': self._status,
			'starting_time': self.starting_time,
			'ending_time': self.ending_time,
			'elapsed_ms': self.get_elapsed(unit='ms'),
			**evaluation
		}

	def do(self, data_namespace, worker_id):
		"""
		:type data_namespace: Namespace
		:type worker_id: int or str
		"""
		try:
			self.start()

			estimator = self.estimator_class(**self.estimator_arguments)
			training_data = getattr(data_namespace, get_training_data_id(data_id_prefix=self.data_id_prefix))
			training_x = training_data[self.x_columns]
			training_y = training_data[self.y_column]
			estimator.fit(X=training_x, y=training_y)

			test_data = getattr(data_namespace, get_test_data_id(data_id_prefix=self.data_id_prefix))
			test_x = test_data[self.x_columns]
			actual = test_data[self.y_column]
			predicted = estimator.predict(test_x)
			self.evaluate(actual=actual, predicted=predicted)

			self.end(worker_id=worker_id)
		except Exception as error:
			self.set_error(error=error)
