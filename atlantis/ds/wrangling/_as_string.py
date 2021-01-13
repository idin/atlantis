from numpy import where
def as_string(x):
	return where(x.isnull(), x, x.astype(str))