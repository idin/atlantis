from pandas import DataFrame
from sklearn.linear_model import LinearRegression, LogisticRegression
from multiprocess.managers import Namespace
from ._Task import Task


def worker(worker_id, data_namespace, to_do, doing, done, proceed, status, evaluation_function):
	"""
	:type worker_id: int or str
	:type data_namespace: Namespace
	:type to_do: list[Task]
	:type doing: dict[int or str, Task]
	:type done: list[Task]
	:type proceed: dict[str, bool]
	:type status: dict[str, str]
	:type evaluation_function: callable

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

			task.start()
			estimator = task.estimator_class(**task.kwargs)
			training_data = getattr(data_namespace, f'{task.data_id}_training')
			training_x = training_data.drop(columns=task.y_column)
			training_y = training_data[task.y_column]
			estimator.fit(X=training_x, y=training_y)

			test_data = getattr(data_namespace, f'{task.data_id}_test')
			test_x = test_data.drop(columns=task.y_column)
			actual = test_data[task.y_column]
			predicted = estimator.predict(test_x)

			task.evaluation = evaluation_function(actual=actual, predicted=predicted)
			task.end(worker_id=worker_id)

		except Exception as error:
			task.set_error(error=error)

		del doing[worker_id]
		done.append(task)

	status[worker_id] = 'ended'
