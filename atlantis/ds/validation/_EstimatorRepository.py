from sklearn.linear_model import LinearRegression, LogisticRegression
from ...collections import create_grid
from ...hash import hash_object


class EstimatorGrid:
	def __init__(self, estimator, parameters=None):
		"""
		:type estimator: LogisticRegression or LinearRegression
		:type parameters: dict
		"""
		self._estimator = estimator
		self._hash_to_id = {}
		self._id_to_parameters = {}

		if parameters is not None:
			self.append(parameters=parameters)

	def get_available_id(self):
		if len(self._id_to_parameters) == 0:
			return 1
		else:
			max_id = max([i for i in self._id_to_parameters.keys()])

			if len(self._id_to_parameters) == max_id:
				return max_id + 1

			else:
				for i in range(1, max_id + 2):
					if i not in self._id_to_parameters:
						return i

	@property
	def estimator_ids(self):
		"""
		:rtype: list[int]
		"""
		return list(self._id_to_parameters.keys())

	def __contains__(self, item):
		return item in self._id_to_parameters

	@property
	def Estimator(self):
		"""
		:rtype: type
		"""
		return self._estimator

	@property
	def name(self):
		return self.Estimator.__name__

	def append(self, parameters):
		"""
		adds a dictionary of parameters if they don't exist
		:type parameters: dict
		"""
		grid = create_grid(dictionary=parameters)

		for dictionary in grid:
			hash_key = hash_object(dictionary)

			if hash_key not in self._hash_to_id:
				new_id = self.get_available_id()
				self._hash_to_id[hash_key] = new_id
				self._id_to_parameters[new_id] = dictionary

	@property
	def estimator_dictionaries(self):
		"""
		:rtype: list[dict]
		"""
		return [
			{'estimator': self.Estimator, 'parameters': parameters, 'id': f'{self.name}_{key}'}
			for key, parameters in self._id_to_parameters.items()
		]

	def __add__(self, other):
		"""
		:type other: EstimatorGrid
		:rtype: EstimatorGrid
		"""
		result = EstimatorGrid(estimator=self.Estimator)

		for parameters in self._id_to_parameters.values():
			result.append(parameters=parameters)

		for parameters in other._id_to_parameters.values():
			result.append(parameters=parameters)

		return result


class EstimatorRepository:
	def __init__(self, estimator=None, parameters=None):
		self._estimator_grids_dictionary = {}
		if estimator is not None and parameters is not None:
			self.append(estimator=estimator, parameters=parameters)

	def __contains__(self, item):
		"""
		:type item: tuple or str
		"""
		if isinstance(item, str):
			return item in self.estimator_grids
		else:
			estimator_name, estimator_id = item
			if estimator_name not in self.estimator_grids:
				return False
			else:
				return estimator_id in self.estimator_grids[estimator_name]

	def append(self, estimator, parameters):
		"""
		:type estimator: type
		:type parameters: dict
		"""
		class_name = estimator.__name__

		if class_name not in self._estimator_grids_dictionary:
			self._estimator_grids_dictionary[class_name] = EstimatorGrid(estimator=estimator)

		self._estimator_grids_dictionary[class_name].append(parameters=parameters)

	@property
	def estimator_grids(self):
		"""
		:rtype: list[EstimatorGrid]
		"""
		return list(self._estimator_grids_dictionary.values())

	@property
	def estimator_dictionaries(self):
		"""
		:rtype: list[dict]
		"""
		return [
			dictionary
			for grid in self.estimator_grids
			for dictionary in grid.estimator_dictionaries
		]

	def __add__(self, other):
		"""
		:type other: EstimatorRepository
		"""
		result = EstimatorRepository()

		for database in self.estimator_grids:
			result.estimator_grids[database.name] = database

		for database in other.estimator_grids:
			if database.name in result.estimator_grids:
				result.estimator_grids[database.name] += database
			else:
				result.estimator_grids[database.name] = database

		return result
