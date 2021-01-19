from ._EstimatorRepository import EstimatorRepository
from ._TrainingTestContainer import TrainingTestContainer
from ._ValidationContainer import ValidationContainer


class EstimatorRace:
	def __init__(self, num_jobs=1):
		self._estimator_repository = None
		self._training_test_containers = []
		self._performances = {}
		self._num_jobs = num_jobs

	@property
	def estimator_repository(self):
		"""
		:rtype: EstimatorRepository or NoneType
		"""
		return self._estimator_repository

	def append_estimator_repository(self, estimator_repository):
		"""
		:type estimator_repository: EstimatorRepository
		"""
		if self._estimator_repository is None:
			self._estimator_repository = estimator_repository
		else:
			self._estimator_repository = self._estimator_repository + estimator_repository

	def append_data_container(self, data_container):
		"""
		:type data_container: TrainingTestContainer or ValidationContainer
		"""
		if isinstance(data_container, TrainingTestContainer):
			self._training_test_containers.append(data_container)
		elif isinstance(data_container, ValidationContainer):
			for training_test_container in data_container.folds:
				self._training_test_containers.append(training_test_container)

	def append_model(self, model, parameters):
		"""
		:type model: LogisticRegression or LinearRegression
		:type parameters: dict
		"""
		if self._estimator_repository is None:
			self._estimator_repository = EstimatorRepository(estimator=model, estimator_arguments=parameters)

		else:
			self.estimator_repository.append(estimator=model, estimator_arguments=parameters)

