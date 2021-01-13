class Sampler:
	def __init__(self, sample_size = None, sample_ratio = None, num_splits = 1):
		self.sample_size = sample_size
		self.sample_ratio = sample_ratio
		self.num_splits = num_splits


class Splitter(Sampler):
	def __init__(self, sample_size = None, sample_ratio = 0.2):
		pass
