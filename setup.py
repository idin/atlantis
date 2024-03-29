from setuptools import setup, find_packages


def readme():
	with open('./README.md') as f:
		return f.read()


setup(
	name='atlantis',
	version='2021.1.3.5',
	description='Python library for simplifying slicing science',
	long_description=readme(),
	long_description_content_type='text/markdown',
	url='https://github.com/idin/atlantis',
	author='Idin',
	author_email='py@idin.ca',
	license='MIT',
	packages=find_packages(exclude=("jupyter", ".idea", ".git", "data_files")),
	install_requires=['base32hex', 'geopy', 'pandas', 'joblib', 'numpy', 'sklearn', 'multiprocess'],
	package_data={'atlantis': ['data_files/*.pickle']},
	python_requires='~=3.6',
	zip_safe=False
)
