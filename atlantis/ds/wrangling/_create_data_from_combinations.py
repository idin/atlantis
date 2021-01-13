from pandas import DataFrame


def create_data_from_combinations(**kwargs):
	"""
	:rtype: DataFrame
	"""
	data_frames = [
		DataFrame({name: list(set(values))})
		for name, values in kwargs.items()
	]
	result = None
	for data in data_frames:
		data['_id_'] = 1
		if result is None:
			result = data
		else:
			result = result.merge(right=data, on='_id_', how='left')
	return result.drop(columns='_id_')
