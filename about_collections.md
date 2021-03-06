# *collections*
This module of the package [***atlantis***](README.md) helps with working with collections.

## *`flatten`*
```python
from atlantis.collections import flatten
flatten([1, 2, [3, 4, [5, 6], 7], 8])
```
returns: `[1, 2, 3, 4, 5, 6, 7, 8]`

## *`List`*
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