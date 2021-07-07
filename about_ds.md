# *ds*
This module of the package [***atlantis***](README.md) provides 
data science tools for:
- data wrangling, 
- validation, 
- tuning,
- sampling, 
- evaluation,
- clustering, and 
- parallel processing of machine learning models.

## Clustering

### *KMeans* Clustering
I have used the `KMeans` class from both *sklearn* and that of *pyspark* and was frustrated 
by two problems: (a) even though the two classes do exactly the same thing their interfaces
are vastly different and (b) some of the simplest operations are very hard to do with 
both classes. I solved this problem by creating my own `KMeans` class that is a wrapper 
aroung both of those classes and uses the appropriate one automatically without 
complicating it for the data scientist programmer. 

### `Elbow`