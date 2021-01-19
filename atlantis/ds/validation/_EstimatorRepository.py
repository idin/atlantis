from sklearn.linear_model import LinearRegression, LogisticRegression
from ...collections import create_grid
from ...hash import hash_object


class EstimatorGrid:
	def __init__(self, estimator, estimator_arguments=None):
		"""
		:type estimator: LogisticRegression or LinearRegression
		:type estimator_arguments: dict
		"""
		self._estimator = estimator
		self._hash_to_id = {}
		self._id_to_arguments = {}

		if estimator_arguments is not None:
			self.append(estimator_arguments=estimator_arguments)

	def get_available_id(self):
		if len(self._id_to_arguments) == 0:
			return 1
		else:
			max_id = max([i for i in self._id_to_arguments.keys()])

			if len(self._id_to_arguments) == max_id:
				return max_id + 1

			else:
				for i in range(1, max_id + 2):
					if i not in self._id_to_arguments:
						return i

	@property
	def estimator_ids(self):
		"""
		:rtype: list[int]
		"""
		return list(self._id_to_arguments.keys())

	def __contains__(self, item):
		return item in self._id_to_arguments

	@property
	def estimator(self):
		"""
		:rtype: type
		"""
		return self._estimator

	@property
	def name(self):
		return self.estimator.__name__

	def append(self, estimator_arguments):
		"""
		adds a dictionary of arguments if they don't exist
		:type estimator_arguments: dict
		"""
		grid = create_grid(dictionary=estimator_arguments)

		for dictionary in grid:
			hash_key = hash_object(dictionary)

			if hash_key not in self._hash_to_id:
				new_id = self.get_available_id()
				self._hash_to_id[hash_key] = new_id
				self._id_to_arguments[new_id] = dictionary

	@property
	def estimator_dictionaries(self):
		"""
		:rtype: list[dict]
		"""
		return [
			{'estimator': self.estimator, 'estimator_arguments': estimator_arguments, 'id': f'{self.name}_{key}'}
			for key, estimator_arguments in self._id_to_arguments.items()
		]

	def __add__(self, other):
		"""
		:type other: EstimatorGrid
		:rtype: EstimatorGrid
		"""
		result = EstimatorGrid(estimator=self.estimator)

		for estimator_arguments in self._id_to_arguments.values():
			result.append(estimator_arguments=estimator_arguments)

		for estimator_arguments in other._id_to_arguments.values():
			result.append(estimator_arguments=estimator_arguments)

		return result


class EstimatorRepository:
	def __init__(self, estimator=None, estimator_arguments=None):
		self._estimator_grids_dictionary = {}
		if estimator is not None and estimator_arguments is not None:
			self.append(estimator=estimator, estimator_arguments=estimator_arguments)

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

	def append(self, estimator, estimator_arguments=None):
		"""
		:type estimator: type
		:type estimator_arguments: dict
		"""
		estimator_arguments = estimator_arguments or {}
		class_name = estimator.__name__

		if class_name not in self._estimator_grids_dictionary:
			self._estimator_grids_dictionary[class_name] = EstimatorGrid(estimator=estimator)

		self._estimator_grids_dictionary[class_name].append(estimator_arguments=estimator_arguments)

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
