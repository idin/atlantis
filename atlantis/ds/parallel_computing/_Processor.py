import multiprocess
from pandas import DataFrame
from time import sleep

from ...time import get_elapsed, get_now
from ...time.progress import ProgressBar

from ..validation import Validation
from ..validation import EstimatorRepository

from ._worker import worker
from ._TimeEstimate import MissingTimeEstimate
from ._Project import Project
from ._LearningProject import LearningProject


class Processor:
	def __init__(self, time_unit='ms'):
		self._processes = {}
		self._manager = multiprocess.Manager()
		self._data_namespace = self._manager.Namespace()
		self._data_columns = self._manager.Namespace()
		self._data_shapes = self._manager.Namespace()
		self._estimators = {}

		self._to_do = self._manager.list()
		self._doing = self._manager.dict()
		self._done = self._manager.list()
		self._processed = []
		self._errors = []
		self._proceed_worker = self._manager.dict()
		self._worker_status = self._manager.dict()
		self._projects = {}

		self._tasks_by_id = {}
		self._time_unit = time_unit
		self._worker_id_counter = 0

	@property
	def data_ids(self):
		"""
		:rtype list[str]
		"""
		return list(self._data_namespace.keys())

	def get_data(self, data_id):
		"""
		:type data_id: int or str
		:rtype: DataFrame
		"""
		return getattr(self._data_namespace, data_id)

	def get_data_columns(self, data_id):
		"""
		:type data_id: int or str
		:rtype: list
		"""
		return getattr(self._data_columns, data_id)

	def get_data_shape(self, data_id):
		"""
		:type data_id: int or str
		:rtype: tuple
		"""
		return getattr(self._data_shapes, data_id)

	def generate_worker_id(self):
		self._worker_id_counter += 1
		return f'worker_{self._worker_id_counter}'

	@property
	def projects(self):
		"""
		:rtype: dict[str, Project or LearningProject]
		"""
		return self._projects

	def add_project(self, project):
		"""
		:type project: Project
		"""
		if project.name in self._projects:
			raise RuntimeError(f'project {project.name} already exists.')
		self._projects[project.name] = project
		project._processor = self

	def add_data(self, data_id, data, overwrite=False):
		"""
		:type data_id: int or str
		:type data: DataFrame
		:type overwrite: bool
		"""
		if hasattr(self._data_namespace, data_id):
			if not overwrite:
				raise ValueError(f'data {data_id} already exists in the processor')
		setattr(self._data_namespace, data_id, data)
		setattr(self._data_columns, data_id, list(data.columns))
		setattr(self._data_shapes, data_id, data.shape)

	def add_worker(self):
		"""
		:rtype: multiprocess.Process
		"""
		worker_id = self.generate_worker_id()
		process = multiprocess.Process(
			target=worker,
			kwargs={
				'worker_id': worker_id,
				'data_namespace': self._data_namespace,
				'to_do': self._to_do,
				'doing': self._doing,
				'done': self._done,
				'proceed': self._proceed_worker,
				'status': self._worker_status
			}
		)
		self._processes[worker_id] = process
		process.start()
		return process

	def add_workers(self, num_workers):
		"""
		:type num_workers: int
		"""
		self.process_done_tasks()
		for i in range(num_workers):
			self.add_worker()

	def create_learning_project(
			self, name, y_column, problem_type, x_columns=None, time_unit='ms',
			evaluation_function=None, main_metric=None, lowest_is_best=None, best_score=None,
			scoreboard=None
	):
		"""

		:type 	name: str
		:type 	y_column: strx

		:type 	problem_type: str
		:param 	problem_type: either regression or classification

		:type 	x_columns: str

		:type 	time_unit: str
		:param 	time_unit: s, ms, etc.


		:type 	evaluation_function: callable
		:param 	evaluation_function: 	a function that gets predicted and actual and
										produces a dictionary of values such as {'rmse': ...} for regression
										or {'f1_score': ...} for classification

		:param 	main_metric: 	the main metric used for comparison, it should exist as one of the keys
								in the result produced by evaluation_function

		:param 	lowest_is_best: usually True for regression (unless a weird metric is used) and False for classification
		:param 	best_score: usually 0 for regression and 1 for classification

		:type 	scoreboard: Scoreboard
		:param 	scoreboard: a Scoreboard object that keeps score of all estimators, can be added later too

		:rtype: LearningProject
		"""
		project = LearningProject(
			processor=self,
			name=name, y_column=y_column, problem_type=problem_type, x_columns=x_columns, time_unit=time_unit,
			evaluation_function=evaluation_function, main_metric=main_metric, lowest_is_best=lowest_is_best,
			best_score=best_score, scoreboard=scoreboard
		)
		return project

	def load_to_do_tasks(self, project_name=None, num_tasks=None, echo=True, **kwargs):
		"""

		:param project_name: name of the project
		:type  project_name: str

		:param num_tasks: number of tasks to load from the project
		:type  num_tasks: int

		:param echo:
		:type  echo: bool

		:return:
		"""
		if project_name is None:
			for project_name in self.projects.keys():
				self.load_to_do_tasks(project_name=project_name, num_tasks=num_tasks, echo=echo, **kwargs)

		self.process_done_tasks()
		project = self.projects[project_name]
		project.produce_tasks(ignore_error=True, echo=echo)
		if num_tasks is not None:
			if num_tasks > project.num_to_do_tasks:
				# fill to do list in project
				num_of_new_tasks = num_tasks - project.num_to_do_tasks

				# but cannot get more than what is available
				num_of_new_tasks = min(num_of_new_tasks, project.num_new_tasks)

				project.fill_to_do_list(num_tasks=num_of_new_tasks, **kwargs)

		loaded_count = 0
		while project.num_to_do_tasks > 0:
			task = project.pop_to_do_task()
			self._to_do.append(task)
			loaded_count += 1

		if echo:
			print(f'{loaded_count} loaded from project {project_name}')

	def process_done_tasks(self, ignore_errors=False, echo=True):
		processed_count = {}
		while True:
			try:
				task = self._done.pop(0)

				if task.status == 'done':
					self.projects[task.project_name].add_time_estimate(task=task)
					project = self.projects[task.project_name]
					project.add_done_task(task=task)
					if task.project_name not in processed_count:
						processed_count[task.project_name] = 1
					else:
						processed_count[task.project_name] += 1
				elif task.status == 'error':
					if not ignore_errors:
						raise task._error
				else:
					raise RuntimeError(f'do not know what to do with status: {task.status}')

				self._processed.append(task)

			except IndexError:
				break
		if echo:
			for project_name, number in processed_count.items():
				print(f'{processed_count} tasks from project {project_name} processed.')

	def get_time_estimate(self, task):
		return self.projects[task.project_name].get_time_estimate(task=task)

	def count_to_do(self):
		return len(self._to_do) + len(self._doing)

	def count_done(self):
		return len(self._processed) + len(self._done)

	def get_to_do_time(self):
		if self.count_to_do() == 0:
			return 0

		total = 0
		for task in self._doing.values():
			estimate = self.get_time_estimate(task=task)
			if estimate == MissingTimeEstimate():
				return estimate
			total += estimate

		for task in self._to_do:
			estimate = self.get_time_estimate(task=task)
			if estimate == MissingTimeEstimate():
				return estimate
			total += self.get_time_estimate(task=task)

		return total

	def get_done_time(self):
		total = 0

		for task in self._processed:
			estimate = self.get_time_estimate(task=task)
			if estimate == MissingTimeEstimate():
				return estimate
			total += estimate

		for task in self._done:
			estimate = self.get_time_estimate(task=task)
			if estimate == MissingTimeEstimate():
				return estimate
			total += estimate

		return total

	def get_worker_count_string(self):
		active = 0
		idle = 0
		terminated_or_ended = 0
		for worker_status in self._worker_status.values():
			if worker_status in {'started', 'active'}:
				active += 1
			elif worker_status == 'idle':
				idle += 1
			elif worker_status in {'ended', 'terminated'}:
				terminated_or_ended += 1
		result = []
		if active > 0:
			result.append(f'{active} active')
		if idle > 0:
			result.append(f'{idle} idle')
		if terminated_or_ended > 0:
			result.append(f'{terminated_or_ended} ended')

		return ' '.join(result)

	def get_worker_count(self):
		return len(self._processes)

	def _update_progress_bay_by_count(self, progress_bar, next_line):
		to_do_count = self.count_to_do()
		done_count = self.count_done()
		progress_bar.set_total(to_do_count + done_count)
		progress_bar.show(
			amount=done_count,
			text=f'tasks: {done_count} / {to_do_count + done_count} | workers: {self.get_worker_count_string()}',
			next_line=next_line
		)
		return to_do_count

	def _update_progress_bar(self, progress_bar, next_line):
		self.process_done_tasks(echo=False)

		# if there is any time estimate use time to show progress

		# try with time:
		to_do_time = self.get_to_do_time()
		if to_do_time == MissingTimeEstimate():
			return self._update_progress_bay_by_count(progress_bar=progress_bar, next_line=next_line)

		done_time = self.get_done_time()
		if done_time == MissingTimeEstimate():
			return self._update_progress_bay_by_count(progress_bar=progress_bar)

		progress_bar.set_total(to_do_time + done_time)
		to_do_count = self.count_to_do()
		done_count = self.count_done()
		progress_bar.show(
			amount=done_time,
			text=f'tasks: {done_count} / {to_do_count + done_count} | workers: {self.get_worker_count_string()}',
			next_line=next_line

		)
		return to_do_count

	def show_progress(self, time_limit=None, time_unit='s'):
		if len(self._processes) == 0:
			raise RuntimeError('there are no workers')
		start_time = get_now()
		progress_bar = ProgressBar(total=100)
		progress_bar.show(amount=0)
		try:
			while True:
				to_do_count = self._update_progress_bar(progress_bar=progress_bar, next_line=False)
				if to_do_count == 0:
					progress_bar.set_total(total=100)
					progress_bar.show(amount=100, text=f'workers: {self.get_worker_count_string()}')
					break
				if time_limit is not None and get_elapsed(start=start_time, unit=time_unit) > time_limit:
					print()
					break
				sleep(0.1)

		except KeyboardInterrupt:
			self._update_progress_bar(progress_bar=progress_bar, next_line=True)

	@property
	def worker_status_table(self):
		return DataFrame.from_records([
			{'id': worker_id, 'status': worker_status}
			for worker_id, worker_status in self._worker_status.items()
		])

	@property
	def tasks(self):
		"""
		:rtype: list[TrainingTestTask]
		"""
		d = {}
		for task in self._processed:
			d[task.name] = task
		for task in self._done:
			d[task.name] = task
		for task in self._doing.values():
			d[task.name] = task
		for task in self._to_do:
			d[task.name] = task
		return list(d.values())

	@property
	def task_table(self):
		return DataFrame.from_records([
			task.record
			for task in self.tasks
		])

	def stop(self, worker_id=None):
		if worker_id is not None:
			self._proceed_worker[worker_id] = False

		else:
			for _worker_id in self._processes.keys():
				self.stop(worker_id=_worker_id)

		return self._done

	def terminate(self, worker_id=None, echo=1):
		if worker_id is not None:
			if worker_id not in self._processes:
				raise KeyError(f'worker {worker_id}')

			self._proceed_worker[worker_id] = False
			self._processes[worker_id].terminate()
			self._to_do.append(self._doing[worker_id])
			del self._doing[worker_id]
			del self._processes[worker_id]
			if self._worker_status[worker_id] != 'ended':
				self._worker_status[worker_id] = 'terminated'
				if echo:
					print(f'worker {worker_id} terminated!')
			else:
				if echo:
					print(f'worker {worker_id} already ended.')

		else:
			worker_ids = list(self._processes.keys())
			for _worker_id in worker_ids:
				self.terminate(worker_id=_worker_id, echo=echo)

		return self._done
