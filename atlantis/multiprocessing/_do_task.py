from time import sleep
from ._WorkerReport import WorkerReport


def do_task(
		to_do_queue, done_queue, workers_doing, tasks_status, keep_workers_alive,
		worker_reports, worker_id,
		cpu_usage, max_cpu_count,
		empty_count_limit, max_sleep_time,
		sleep_time=0.1, echo=0
):
	"""
	:type to_do_queue: multiprocessing.Queue[Task]
	:type done_queue: multiprocessing.Queue[Outcome]
	:type workers_doing: dict
	:type tasks_status: dict
	:type keep_workers_alive: multiprocessing.Value
	:type worker_reports: dict
	:type worker_id: str or int
	:type cpu_usage: multiprocessing.Value
	:type max_cpu_count: int
	:type empty_count_limit: int
	:type max_sleep_time: float
	:type sleep_time: float
	:type echo: bool or int
	:rtype:
	"""

	report = WorkerReport(worker_id=worker_id)

	def print_if_echo(x):
		if echo:
			print(x)

	empty_count = 0
	# sleep(sleep_time)
	while max_cpu_count > cpu_usage.value:
		try:
			print_if_echo(f'{worker_id} getting task')
			task = to_do_queue.get_nowait()
			empty_count = 0
		except Exception:
			print_if_echo(f'{worker_id} queue empty')
			empty_count += 1
			if keep_workers_alive.value == 0:
				break
			if empty_count_limit is not None and empty_count >= empty_count_limit:
				break
			else:
				sleep(min(sleep_time * 2 ** empty_count, max_sleep_time))
		else:
			print_if_echo(f'{worker_id} got task {task.id}')

			tasks_status[task.id] = 'started'
			workers_doing[worker_id] = task.id

			print_if_echo(f'{worker_id} doing task {task.id}')
			cpu_usage.value += task.cpu_count
			outcome = task.do(worker_id=worker_id)
			cpu_usage.value -= task.cpu_count
			print_if_echo(f'{worker_id} did task {task.id}')

			report.add_task_id(task_id=task.id)

			print_if_echo(f'{worker_id} putting result of {task.id}')
			done_queue.put(outcome)

			workers_doing[worker_id] = None
			tasks_status[task.id] = 'done'

			print_if_echo(f'{worker_id} waiting')
			sleep(sleep_time)

	print_if_echo(f'{worker_id} out of the loop')
	report.end()
	print_if_echo(f'{worker_id} putting report')
	worker_reports[worker_id] = report
	return True
