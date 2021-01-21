from atlantis.ds.synthetic_data import create_data
from atlantis.ds.validation import CrossValidation
from atlantis.ds.validation import EstimatorRepository
from atlantis.ds.parallel_computing import Processor

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

repository = EstimatorRepository()
repository.append(RandomForestRegressor, {'n_estimators': 100, 'n_jobs': 1, 'max_depth': [6, 12, 24]})
repository.append(Lasso, {'alpha': [0, 0.1, 1, 10]})
repository.append(Lasso, {'alpha': [1, 10, 20]})
repository.append(LinearRegression)

data = create_data(num_rows=10000, num_x_columns=100, noise=2)
cv = CrossValidation(num_splits=5)

processor = Processor()
processor.add_workers(num_workers=8)

project = processor.create_cross_validation_project(name='example', y_column='y', problem_type='regression')
project.add_estimator_repository(repository=repository)
project.add_validation(data=data, validation=cv, random_state=42)
display(project)

project.send_to_do(num_tasks=10)

processor.show_progress()
