from time import sleep
import multiprocess

from ._WorkerReport import SupervisorReport
from ._do_task import do_task


def supervise(
		to_do_queue, done_queue, processed, incomplete_task_ids,
		workers, workers_doing, worker_reports, supervisor_id,
		cpu_usage, max_cpu_count,
		worker_id_counter, supervisor_alive, supervisor_reports,
		manager,
		sleep_time=0.1, worker_sleep_time=0.1, echo=0
):
	"""
	:type to_do_queue: multiprocessing.Queue[Task]
	:type done_queue: multiprocessing.Queue[TaskResult]
	:type processed: list[TaskResult]
	:type workers: dict[str, multiprocessing.Process]
	:type workers_doing: dict
	:type worker_reports: dict
	:type incomplete_task_ids: dict
	:type supervisor_id: str or int
	:type cpu_usage: multiprocessing.Value
	:type max_cpu_count: int
	:type worker_id_counter: int
	:type supervisor_alive: multiprocessing.Value
	:type supervisor_reports: dict
	:type sleep_time: float
	:type worker_sleep_time: float
	:type echo: bool or int
	"""
	report = SupervisorReport(supervisor_id=supervisor_id)
	report.log(f'supervisor {supervisor_id} started')

	worker_id_counter = 0

	while supervisor_alive.value == 1:

		# process done queue
		while True:
			try:
				result = done_queue.get_nowait()
			except:
				break
			else:
				processed.append(result)
				report.log(f'task {result.task_id} processed')

		# remove dead workers
		dead_worker_ids = set()
		if workers is not None:
			for worker_id, worker in workers.items():
				if not worker.is_alive():
					dead_worker_ids.add(worker_id)
					if worker in workers_doing:
						incomplete_task_id = workers_doing[worker_id]
						if incomplete_task_id is not None:
							incomplete_task_ids[incomplete_task_id] = 1
							report.log(f'incomplete task {incomplete_task_id} captured')
			for dead_worker_id in dead_worker_ids:
				del worker_id[dead_worker_id]
				report.log(f'dead worker {dead_worker_id} removed')

		# add workers as needed
		if workers is not None:
			while not to_do_queue.empty() and cpu_usage.value < max_cpu_count:

				# add worker
				new_worker_id = worker_id_counter.value
				worker_id_counter.value += 1
				process = multiprocess.Process(
					target=do_task,
					kwargs={
						'to_do_queue': to_do_queue,
						'done_queue': done_queue,
						'workers_doing': workers_doing,
						'worker_reports': worker_reports,
						'worker_id': new_worker_id,
						'cpu_usage': cpu_usage,
						'max_cpu_count': max_cpu_count,
						'sleep_time': worker_sleep_time,
						'echo': echo
					}
				)
				workers[new_worker_id] = process
				process.start()
				report.log(f'new worker {new_worker_id} added')
				sleep(sleep_time)

		sleep(sleep_time)

	report.log(f'supervisor {supervisor_id} ends')
	report.end()
	supervisor_reports.append(report)
	return True
