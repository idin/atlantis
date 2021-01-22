from multiprocess.managers import Namespace
from time import sleep

def worker(worker_id, namespace, to_do, doing, done, proceed, status):
	"""
	:type worker_id: int or str
	:type namespace: Namespace
	:type to_do: list[TrainingTestTask]
	:type doing: dict[int or str, Task]
	:type done: list[Task]
	:type proceed: dict[str, bool]
	:type status: dict[str, str]

	each item in the queue is tuple or list that has:
	estimator_id, data_id, estimator class, dictionary of kwargs, training, test

	"""
	status[worker_id] = 'started'
	if worker_id in proceed:
		error = ValueError(f'{worker_id} already exists in proceed')
		status[worker_id] = f'error: {error}'
		raise error
	else:
		proceed[worker_id] = True

	while proceed[worker_id]:
		try:
			task = to_do.pop(0)
			doing[worker_id] = task
			status[worker_id] = 'active'

		except IndexError:
			status[worker_id] = 'idle'
			continue

		try:
			task.do(namespace=namespace, worker_id=worker_id)

		except Exception as error:
			task.add_error(error=error)
		sleep(0.1)

		del doing[worker_id]
		done.append(task)

	status[worker_id] = 'ended'
