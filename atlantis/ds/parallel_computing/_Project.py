from collections import OrderedDict
from ...time.progress import ProgressBar
from ._TimeEstimate import TimeEstimate, MissingTimeEstimate
from ._Task import Task
import warnings


class Project:
	def __init__(self, name, processor, time_unit='ms'):
		"""
		:type name: str
		:type time_unit: str
		"""
		self._name = name
		self._time_unit = time_unit
		self._time_estimates = {}
		self._pre_to_do = OrderedDict()
		self._to_do = OrderedDict()

		self._being_done_ids = set()

		self._done = OrderedDict()
		self._processor = processor
		processor.add_project(project=self)

	def __repr__(self):
		lines = [
			f'Name: {self.name}',
			'',
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
			self.time_estimates[task.time_estimate_id] = TimeEstimate()
		self.time_estimates[task.time_estimate_id].append(task.get_elapsed(unit=self._time_unit))

	def get_time_estimate(self, task):
		if task.project_name != self.name:
			raise RuntimeError(f'task.problem_id = {task.project_name} does not match problem_id = {self.name}')

		if task.is_done():
			return task.get_elapsed(unit=self._time_unit)

		if task.estimator_name in self.time_estimates:
			return self.time_estimates[task.time_estimate_id].get_mean()

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
	def processor(self):
		"""
		:rtype: atlantis.ds.parallel_computing.Processor
		"""
		return self._processor

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

	def produce_tasks(self, **kwargs):
		raise NotImplementedError(f'this method should be implemented for class {self.__class__}')

	def fill_to_do_list(self, num_tasks, echo=True, **kwargs):
		raise NotImplementedError(f'this method should be implemented for class {self.__class__}')

	def send_to_do(self, num_tasks=None, echo=True, **kwargs):
		self._processor.receive_to_do(
			project_name=self.name, num_tasks=num_tasks, echo=echo, process_done_tasks=True, **kwargs
		)

	def do(self, num_tasks=None, echo=True, disable_warnings=True, **kwargs):
		if not self._all_tasks_produced:
			self.produce_tasks(ignore_error=True, echo=echo)
		num_tasks = min(num_tasks, self.to_do_count + self.new_count)
		progress_bar = ProgressBar(total=num_tasks, echo=echo)

		num_done = 0
		to_do_count = self.to_do_count
		for i in range(to_do_count):
			progress_bar.show(amount=num_done, text=f'done: {num_done} / {to_do_count} (to-do)')
			task = self.pop_to_do()
			if disable_warnings:
				with warnings.catch_warnings():
					warnings.simplefilter('ignore')
					task.do(namespace=self.processor.namespace, worker_id='main')
			else:
				task.do(namespace=self.processor.namespace, worker_id='main')
			self.add_done_task(task=task)
			num_done += 1
		progress_bar.show(amount=num_done, text=f'done: {num_done} / {to_do_count} (to-do)')

		while num_tasks > num_done:
			progress_bar.show(amount=num_done, text=f'done: {num_done} / {num_tasks} (all tasks)')
			self.fill_to_do_list(num_tasks=1, echo=False, **kwargs)
			task = self.pop_to_do()
			if disable_warnings:
				with warnings.catch_warnings():
					warnings.simplefilter('ignore')
					task.do(namespace=self.processor.namespace, worker_id='main')
			else:
				task.do(namespace=self.processor.namespace, worker_id='main')
			self.add_done_task(task=task)
			num_done += 1
		progress_bar.show(amount=num_done, text=f'done: {num_done} / {num_tasks} (all tasks)')
