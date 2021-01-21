def namespace_has(namespace, obj_type, obj_id):
	return hasattr(namespace, f'{obj_type}_{obj_id}')


def namespace_has_data(namespace, data_id):
	return namespace_has(namespace=namespace, obj_type='data', obj_id=data_id)


def add_obj_to_namespace(namespace, obj_type, obj_id, obj, overwrite=False):
	if namespace_has(namespace=namespace, obj_type=obj_type, obj_id=obj_id):
		if not overwrite:
			raise ValueError(f'{obj_type} {obj_id} already exists in the namespace')

	setattr(namespace, f'{obj_type}_{obj_id}', obj)


def get_data_from_namespace(namespace, data_id):
	return get_obj_from_namespace(namespace=namespace, obj_type='data', obj_id=data_id)


def get_obj_from_namespace(namespace, obj_type, obj_id):
	return getattr(namespace, f'{obj_type}_{obj_id}')
