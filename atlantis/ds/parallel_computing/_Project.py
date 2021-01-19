from ._TimeEstimate import TimeEstimate, MissingTimeEstimate
from ._Task import Task
from collections import OrderedDict


class Project:
	def __init__(self, name, time_unit='ms', processor=None):
		"""
		:type name: str
		:type time_unit: str
		"""
		self._name = name
		self._time_unit = time_unit
		self._time_estimates = {}
		self._data_ids = set()
		self._estimators = {}

		self._pre_to_do = OrderedDict()
		self._to_do = OrderedDict()

		self._being_done_ids = set()

		self._done = OrderedDict()
		self._processor = None
		if processor is not None:
			processor.add_project(project=self)

	def __repr__(self):
		lines = [
			f'Name: {self.name}',
			'',
			f'estimators: {len(self.estimators)}',
			f'data sets: {len(self._data_ids)}',
			f'tasks: {self.task_count} (to-do: {self.to_do_count}, being done: {self.being_done_count}, done: {self.done_count})'
		]

		return '\n'.join(lines)

	def __str__(self):
		return str(self.name)

	@property
	def name(self):
		return self._name

	@property
	def time_estimates(self):
		"""
		:rtype: dict[str, TimeEstimate]
		"""
		return self._time_estimates

	def add_time_estimate(self, task):
		"""
		:type task: TrainingTestTask
		"""
		if task.estimator_name not in self.time_estimates:
			self.time_estimates[task.estimator_name] = TimeEstimate()
		self.time_estimates[task.estimator_name].append(task.get_elapsed(unit=self._time_unit))

	def get_time_estimate(self, task):
		if task.project_name != self.name:
			raise RuntimeError(f'task.problem_id = {task.project_name} does not match problem_id = {self.name}')

		if task.is_done():
			return task.get_elapsed(unit=self._time_unit)

		if task.estimator_name in self.time_estimates:
			return self.time_estimates[task.estimator_name].get_mean()

		elif len(self.time_estimates) > 0:
			total = 0
			count = 0
			for estimate in self.time_estimates.values():
				total += estimate.mean
				count += 1
			return total / count

		else:
			return MissingTimeEstimate()

	@property
	def data_ids(self):
		"""
		:rtype: set
		"""
		return self._data_ids

	def add_data(self, data_id, processor=None, data=None, overwrite=False):
		"""

		:type 	processor: atlantis.ds.parallel_computing.Processor
		:param 	data_id:
		:param 	data:
		:param 	overwrite:
		:return:
		"""
		if processor is not None:
			processor.add_project(project=self)
		else:
			processor = self._processor

		if data_id in self.data_ids and data is not None:
			if not overwrite:
				raise ValueError(f'data {data_id} already exists in project {self}')

		if data is None:
			if data_id not in processor.data_ids:
				raise KeyError(f'data {data_id} does not exist in the processor')
		else:
			processor.add_data(data_id=data_id, data=data, overwrite=overwrite)

		self._data_ids.add(data_id)

	@staticmethod
	def _get_estimator_name(estimator_class):
		return estimator_class.__name__

	def add_estimator(self, estimator_class, estimator_id, estimator_arguments):
		if not isinstance(estimator_class, type):
			raise TypeError(f'estimator_class is of type {type(estimator_class)}')

		estimator_name = self._get_estimator_name(estimator_class)
		key = estimator_name, estimator_id
		self._estimators[key] = {'class': estimator_class, 'arguments': estimator_arguments}
		return estimator_name, estimator_id

	@property
	def estimators(self):
		"""
		:rtype: dict[(str, int), dict]
		"""
		return self._estimators

	@property
	def new_count(self):
		return len(self._pre_to_do)

	@property
	def to_do_count(self):
		return len(self._to_do)

	@property
	def being_done_count(self):
		return len(self._being_done_ids)

	@property
	def done_count(self):
		return len(self._done)

	@property
	def task_count(self):
		return self.new_count + self.to_do_count + self.being_done_count + self.done_count

	def contains_task(self, task_id):
		if isinstance(task_id, Task):
			raise TypeError('task_id cannot be of type Task!')
		return task_id in self._pre_to_do or task_id in self._to_do or task_id in self._being_done_ids or task_id in self._done

	def _take_from_pre_and_add_to_to_do(self, task_id):
		if task_id in self._pre_to_do:
			task = self._pre_to_do.pop(key=task_id)
			self._to_do[task_id] = task
		else:
			raise KeyError(f'task_id {task_id} does not exist in pre-to-do')

	def pop_to_do(self):
		"""
		:rtype: Task
		"""
		task_id, task = self._to_do.popitem(0)
		if task.id in self._being_done_ids:
			self._to_do[task_id] = task
			raise RuntimeError(f'task {task} already exists in being_done')
		self._being_done_ids.add(task_id)
		return task

	def add_done_task(self, task):
		"""
		:type task: Task
		"""
		if not isinstance(task, Task):
			raise TypeError(f'task should be of type Task but it is of type {type(task)}')
		if task.id not in self._being_done_ids:
			raise KeyError(f'task_id {task.id} does not exist in being_done_ids')
		if task.id in self._done:
			raise KeyError(f'task {task} already exists in done!')

		self.process(task=task)
		self._being_done_ids.remove(task.id)
		self._done[task.id] = task

	def process(self, task):
		raise NotImplementedError(f'this method should be implemented for class {self.__class__}')

	def fill_to_do_list(self, num_tasks, **kwargs):
		raise NotImplementedError(f'this method should be implemented for class {self.__class__}')

	def task_is_done(self, task):
		"""
		:type task: Task
		"""
		if not isinstance(task, Task):
			raise TypeError(f'task should be a Task but it is of type {type(Task)}')

		if task.id not in self.task_ids_of_being_done:
			raise KeyError(f'task_id {task.id} does not exist in being_done')

		if task.id in self.done_tasks:
			raise RuntimeError(f'task {task} is already done!')

		try:
			self.process(task=task)
		except Exception as e:
			self._task_errors[task.id] = task
			raise e

		self.done_tasks[task.id] = task
		self.task_ids_of_being_done.remove(task.id)

	def send_to_do(self, num_tasks=None, echo=True, process_done_tasks=True, **kwargs):
		self._processor.receive_to_do(
			project_name=self.name, num_tasks=num_tasks, echo=echo, process_done_tasks=process_done_tasks, **kwargs
		)
