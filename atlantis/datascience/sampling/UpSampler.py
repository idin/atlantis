import pandas as pd
import math
from random import shuffle

class UpSampler:
	def __init__(self, fill_na=None, max_repeats=None):
		#self._sm = SMOTE(**kwargs)
		self._x_cols = None
		self._y_col = None
		self._fill_na = fill_na
		self._up_sampling_table = None
		self._max_repeats = max_repeats

	def fit(self, X, y):
		self._x_cols = X.columns
		self._y_col = y.name

		if self._fill_na is not None:
			y = y.fillna(self._fill_na)

		x_groups = {name: x for name, x in X.groupby(by=y)}
		#y_groups = {name: x for name, x in y.groupby(by=y)}
		group_names = list(x_groups.keys())
		up_sampling_table = pd.DataFrame({
			'name': group_names,
			'population': [x_groups[name].shape[0] for name in group_names]
		})
		num_groups = len(group_names)
		up_sampling_table['total_pop_according_to_groups'] = up_sampling_table['population'] * num_groups
		self._max_pop = up_sampling_table['total_pop_according_to_groups'].max()
		up_sampling_table['max_population'] = self._max_pop

		self._group_pop = int(self._max_pop / num_groups)
		up_sampling_table['ideal_group_pop'] = self._group_pop
		up_sampling_table['num_repeats_required'] = up_sampling_table['ideal_group_pop'] / up_sampling_table[
			'population']
		self._up_sampling_table = up_sampling_table
		self._repeats = up_sampling_table.set_index('name').to_dict()['num_repeats_required']
		if self._max_repeats is not None:
			self._repeats = {key:min(value, self._max_repeats) for key, value in self._repeats.items()}


	def fit_upsample(self, X, y):
		self.fit(X=X, y=y)
		return self.upsample(X=X, y=y)

	def upsample(self, X, y):
		"""
		:type X: pd.DataFrame
		:type y: pd.Series
		:rtype: tuple
		"""
		order = list(range(X.shape[0]))
		shuffle(order)
		X.index = order
		y.index = order

		if self._fill_na is not None:
			y = y.fillna(self._fill_na)



		x_groups = [
			pd.concat([x.sort_index()]*math.ceil(self._repeats[name])).head(self._group_pop)
			for name, x in X.groupby(by=y)
		]
		y_groups = [
			pd.concat([x.sort_index()] * math.ceil(self._repeats[name])).head(self._group_pop)
			for name, x in y.groupby(by=y)
		]

		return pd.concat(x_groups), pd.concat(y_groups)