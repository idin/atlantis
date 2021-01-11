from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


class PrincipalComponentModel:
	def __init__(self, model, pca):
		"""
		:type model: LinearRegression
		:type pca: PCA
		"""
		self._model = model
		self._pca = pca

	def fit(self, X, y, **kwargs):
		transformed_x = self._pca.fit_transform(X=X, y=y)
		self._model.fit(X=transformed_x, y=y, **kwargs)

	def predict(self, X):
		transformed_x = self._pca.transform(X=X)
		return self._model.predict(X=transformed_x)


class PrincipalComponentModelFactory:
	def __init__(self, model_class):
		self._model_class = model_class

	def compose_pca(self, **kwargs):
		"""
		:param kwargs:
		:rtype: PrincipalComponentModel
		"""
		pca_kwargs = {key[4:]: value for key, value in kwargs.items() if key.startswith('pca_')}
		model_kwargs = {key: value for key, value in kwargs.items() if not key.startswith('pca_')}
		pca = PCA(**pca_kwargs)
		model = self._model_class(**model_kwargs)

		return PrincipalComponentModel(model=model, pca=pca)
