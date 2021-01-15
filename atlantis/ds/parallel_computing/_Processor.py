import multiprocess
from pandas import DataFrame
from time import sleep
import random
from ...time import get_elapsed, get_now
from ...time.progress import ProgressBar
from ._worker import worker
from ._Task import Task
from ._TimeEstimate import MissingTimeEstimate
from ._Project import Project
from ..validation import Scoreboard


class Processor:
	def __init__(self, time_unit='ms'):
		self._processes = {}
		self._manager = multiprocess.Manager()
		self._data_namespace = self._manager.Namespace()
		self._estimators = {}

		self._to_do = self._manager.list()
		self._doing = self._manager.dict()
		self._done = self._manager.list()
		self._processed = []
		self._errors = []
		self._proceed_worker = self._manager.dict()
		self._worker_status = self._manager.dict()
		self._projects = self._manager.dict()
		self._scoreboards = {}

		self._tasks_by_id = {}
		self._time_unit = time_unit
		self._worker_id_counter = 0

	@property
	def projects(self):
		"""
		:rtype: dict[str, Project]
		"""
		return self._projects

	@property
	def scoreboards(self):
		"""
		:rtype: dict[str, Scoreboard]
		"""
		return self._scoreboards

	def generate_worker_id(self):
		self._worker_id_counter += 1
		return f'worker_{self._worker_id_counter}'

	def add_project(self, project_name, problem_type, y_column, evaluation_function=None):
		self._projects[project_name] = Project(
			name=project_name, problem_type=problem_type,
			n_function=evaluation_function,
			y_column=y_column, time_unit=self._time_unit
		)

	def add_data(self, project_name, data_id, training_data, test_data):
		if project_name not in self._projects:
			raise KeyError(f'project "{project_name}" is not defined yet!')
		setattr(self._data_namespace, f'{project_name}_{data_id}_training', training_data)
		setattr(self._data_namespace, f'{project_name}_{data_id}_test', test_data)
		self.projects[project_name].add_data_id(data_id=data_id)
		if project_name in self.scoreboards:
			self.scoreboards[project_name].add_data_id(data_id=data_id)

	def add_estimator(self, project_name, ):
		if project_name not in self._estimators:
			self._estimators[project_name] = {}

		if project_name in self.scoreboards:
			self.scoreboards[project_name].add_estimator(estimator_type=estimator_type, estimator_id=estimator_id)



	def add_all_estimator_data_combination_tasks(self, project_name, num_tasks=None, shuffle=True):
		combinations = self.projects[project_name].get_all_estimator_data_combinations()
		if shuffle:
			random.shuffle(combinations)

		if num_tasks is None:
			num_tasks = len(combinations)
		else:
			num_tasks = min(len(combinations), num_tasks)

		for combination in combinations[:num_tasks]:
			self.add_task(es)



	def add_task(self, estimator_id, estimator_class, kwargs, project_name, data_id):
		if data_id not in self.projects[project_name].data_ids:
			raise KeyError(f'data {data_id} does not exist!')
		task = Task(
			estimator_id=estimator_id, estimator_class=estimator_class, kwargs=kwargs,
			project_name=project_name, data_id=data_id, y_column=self.projects[project_name].y_column
		)
		if task.id in self._tasks_by_id:
			raise ValueError(f'Task {task.id} already exists!')
		self._tasks_by_id[task.id] = task
		self._to_do.append(task)

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
				'status': self._worker_status,
				'projects': self._projects
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

	def process_done_tasks(self):
		while True:
			try:
				task = self._done.pop(0)

				if task.status == 'done':
					self.projects[task.project_name].add_time_estimate(task=task)

				self._processed.append(task)
				project_name = task.project_name
				if project_name in self.scoreboards:
					self.scoreboards[project_name].add_score(
						estimator_type=task.estimator_type,
						estimator_id=task.estimator_id,
						data_id=task.data_id,
						score_dictionary=task.evaluation
					)
			except IndexError:
				break

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

	def _update_progress_bay_by_count(self, progress_bar):
		to_do_count = self.count_to_do()
		done_count = self.count_done()
		progress_bar.set_total(to_do_count + done_count)
		progress_bar.show(
			amount=done_count,
			text=f'tasks: {done_count} / {to_do_count + done_count} | workers: {self.get_worker_count_string()}'
		)
		return progress_bar

	def _update_progress_bar(self, progress_bar):
		self.process_done_tasks()

		# if there is any time estimate use time to show progress

		# try with time:
		to_do_time = self.get_to_do_time()
		if to_do_time == MissingTimeEstimate():
			return self._update_progress_bay_by_count(progress_bar=progress_bar)

		done_time = self.get_done_time()
		if done_time == MissingTimeEstimate():
			return self._update_progress_bay_by_count(progress_bar=progress_bar)

		progress_bar.set_total(to_do_time + done_time)
		progress_bar.show(
			amount=done_time,
			text=f'tasks: {done_time} / {to_do_time + done_time} | workers: {self.get_worker_count_string()}'
		)
		return progress_bar

	def show_progress(self, time_limit=None, time_unit='s'):
		if len(self._processes) == 0:
			raise RuntimeError('there are no workers')
		start_time = get_now()
		progress_bar = ProgressBar(total=100)
		progress_bar.show(amount=0)
		try:
			while True:
				to_do_count = self._update_progress_bar(progress_bar=progress_bar)
				if to_do_count == 0:
					progress_bar.set_total(total=100)
					progress_bar.show(amount=100)
					break
				if time_limit is not None and get_elapsed(start=start_time, unit=time_unit) > time_limit:
					break
				sleep(0.1)

		except KeyboardInterrupt:
			self._update_progress_bar(progress_bar=progress_bar)

	@property
	def worker_status_table(self):
		return DataFrame.from_records([
			{'id': worker_id, 'status': worker_status}
			for worker_id, worker_status in self._worker_status.items()
		])

	@property
	def tasks(self):
		"""
		:rtype: list[Task]
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
