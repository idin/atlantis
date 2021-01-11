from pandas import DataFrame


def _has_dataframe(obj):
	if isinstance(obj, dict):
		for value in obj.values():
			if isinstance(value, DataFrame):
				return True
			if isinstance(value, (dict, list, tuple, set)):
				if _has_dataframe(value):
					return True
		return False
	elif isinstance(obj, (list, tuple, set)):
		for element in obj:
			if isinstance(element, DataFrame):
				return True
			if isinstance(element, (dict, list, tuple, set)):
				if _has_dataframe(element):
					return True
		return False
	elif isinstance(obj, DataFrame):
		return True

	else:
		return False


TAB = ' ' * 4


def _display_dataframe_container(obj, display_function, have_display, obj_has_dataframe, depth=0):

	if obj_has_dataframe is None:
		obj_has_dataframe = _has_dataframe(obj)

	if not obj_has_dataframe:
		print(TAB * depth, obj, sep='')

	else:
		if display_function is None and have_display is None:
			try:
				from IPython.core.display import display
				have_display = True
			except ImportError:
				display = None
				have_display = False
		elif display_function is None and not have_display:
			display = None

		else:
			display = display_function

		if isinstance(obj, DataFrame):
			if have_display:
				display(obj)
			else:
				print(TAB * depth, obj, sep='')

		elif isinstance(obj, dict):
			for key, value in obj.items():
				value_has_dataframe = _has_dataframe(value)

				if value_has_dataframe:
					print(TAB * depth, f'{key}:', sep='')
					_display_dataframe_container(
						obj=value, display_function=display, have_display=have_display,
						obj_has_dataframe=value_has_dataframe, depth=depth + 1
					)
				else:
					print(TAB * depth, f'{key}: {value}')

		elif isinstance(obj, (list, set, tuple)):
			for element in obj:
				_display_dataframe_container(
					obj=element, display_function=display, obj_has_dataframe=None, have_display=have_display,
					depth=depth + 1
				)

		else:
			_display_dataframe_container(
				obj=obj, display_function=display_function, have_display=have_display,
				obj_has_dataframe=None, depth=depth + 1
			)


def display_dataframe_container(obj):
	return _display_dataframe_container(
		obj=obj, display_function=None, obj_has_dataframe=None, have_display=None, depth=0
	)


class JupyterContainer:
	def display(self, p=None):
		display_dataframe_container(self)

	def _repr_pretty_(self, p, cycle):
		if cycle:
			p.text('Validation Dictionary')
		else:
			self.display(p=p)


class JupyterDictionary(JupyterContainer, dict):
	pass


class JupyterList(JupyterContainer, list):
	pass


class JupyterTuple(JupyterContainer, tuple):
	pass


class JupyterSet(JupyterContainer, set):
	pass
