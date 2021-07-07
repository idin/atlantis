# ***Atlantis***
***Atlantis*** is a Python library for simplifying programming with Python for data science.

# Installation
You can just use pip to install Atlantis:

`pip install atlantis`

# Modules

- [***collections***](#collections) helps with working with collections.
- [***colour***](about_colour.md) simplifies using colours.
- [***ds (datascience)***](#ds-data-science) provides tools for:
  - data wrangling, 
  - validation, 
  - tuning,
  - sampling, 
  - evaluation,
  - clustering, and 
  - parallel processing of machine learning models.
- [***functions***](about_functions.md) manages higher order functions.
- [***hash***](about_hash.md) simplifies and standardizes hashing.
- [***text***](about_text.md) makes working with texts and strings easy.
- [***time***](about_time.md) 
  - provides methods for interacting with time and date as well as 
  - progress bars
  
## *collections*
This module of the package [***atlantis***](README.md) helps with working with collections.

### *`flatten`*
```python
from atlantis.collections import flatten
flatten([1, 2, [3, 4, [5, 6], 7], 8])
```
returns: `[1, 2, 3, 4, 5, 6, 7, 8]`

### *`List`*
This class inherits from Python's list class but implements a few 
additional functionalities.

```python
from atlantis.collections import List
l = List(1, 2, 3, 4, 2, [1, 2], [1, 2])
```

Flattening: 
```python
l.flatten()
>>> List: [1, 2, 3, 4, 2, 1, 2, 1, 2]
```

Finding duplicates:
```python
l.get_duplicates()
>>> List: [2, List: [1, 2]]
```
**Note:** the ***list*** elements of a ***List*** automatically get converted to ***Lists***, recursively.

## *ds* (Data Science)
This module provides data science tools for:
- data wrangling, 
- validation, 
- tuning,
- sampling, 
- evaluation,
- clustering, and 
- parallel processing of machine learning models.

### *KMeans* Clustering
I have used the `KMeans` class from both *sklearn* and that of *pyspark* and was frustrated 
by two problems: (a) even though the two classes do exactly the same thing their interfaces
are vastly different and (b) some of the simplest operations are very hard to do with 
both classes. I solved this problem by creating my own `KMeans` class that is a wrapper 
aroung both of those classes and uses the appropriate one automatically without 
complicating it for the data scientist programmer. 

### Clustering Optimization