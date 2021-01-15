from ...time import get_elapsed
from datetime import datetime


class Task:
	def __init__(self, project_name):
		self._project_name = project_name
		self._worker_id = None

	@property
	def starting_time(self):
		return self._starting_time

	@property
	def ending_time(self):
		return self._ending_time

	def start(self):
		self._status = 'started'
		self._starting_time = datetime.now()

	def end(self, worker_id):
		self._ending_time = datetime.now()
		self._status = 'done'
		self._worker_id = worker_id

	def get_elapsed(self, unit='ms'):
		if self._starting_time is not None and self._ending_time is not None:
			return get_elapsed(start=self.starting_time, end=self.ending_time, unit=unit)
		else:
			return None

	def set_error(self, error):
		self._error = error
		self._status = 'error'

	def is_done(self):
		return self._status == 'done'


class TrainingTestTask(Task):
	def __init__(
			self, estimator_id, project_name, data_id, estimator_class, kwargs, y_column
	):
		super().__init__(project_name=project_name)
		if not isinstance(estimator_id, (str, int)):
			raise TypeError('estimator_id should be an int or str')

		if not isinstance(project_name, (str, int)):
			raise TypeError('project_name should be either an int or a str')

		if not isinstance(data_id, (int, str)):
			raise TypeError('data_id should be an int or str')

		if not isinstance(estimator_class, type):
			raise TypeError('estimator_class should be a type')

		if not isinstance(kwargs, dict):
			raise TypeError('kwargs should be an int')

		if not isinstance(y_column, str):
			raise TypeError('y_column should be a str')

		self._estimator_id = estimator_id

		self._data_id = data_id
		self._estimator_class = estimator_class
		self._kwargs = kwargs

		self._y_column = y_column
		self._status = 'new'
		self._evaluation = None
		self._starting_time = None
		self._ending_time = None
		self._error = None


	def __repr__(self):
		return f'Task: {self.id} ({self._status})'

	def __str__(self):
		return repr(self)

	@property
	def id(self):
		return self.estimator_type, self.estimator_id, self.project_name, self.data_id, self.y_column

	@property
	def status(self):
		return self._status

	def __hash__(self):
		return hash(self.id)

	@property
	def estimator_id(self):
		return self._estimator_id

	@property
	def project_name(self):
		return self._project_name

	@property
	def estimator_class(self):
		return self._estimator_class

	@property
	def estimator_type(self):
		return self.estimator_class.__name__

	@property
	def data_id(self):
		return self._data_id

	@property
	def kwargs(self):
		return self._kwargs

	@property
	def y_column(self):
		return self._y_column

	@property
	def evaluation(self):
		return self._evaluation

	@evaluation.setter
	def evaluation(self, evaluation):
		self._evaluation = evaluation

	@property
	def record(self):
		"""
		:rtype: dict
		"""
		evaluation = self.evaluation or {}

		return {
			'project_name': self.project_name,
			'estimator_type': self.estimator_type,
			'estimator_id': self.estimator_id,
			'data_id': self.data_id,
			'worker_id': self._worker_id,
			'status': self._status,
			'starting_time': self.starting_time,
			'ending_time': self.ending_time,
			'elapsed_ms': self.get_elapsed(unit='ms'),
			**evaluation
		}
