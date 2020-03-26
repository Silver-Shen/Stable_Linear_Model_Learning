# Stable_Linear_Model_Learning

The implementation of AAAI2020 paper "Stable Learning via Sample Reweighting"

Example:

```python
# Load data to a numpy array X with shape n (sample) by p (feature)
...
...
...
# Calculate new sample weights based on decorrelation operator
weights = decorrelation(X)

# Incorporate sample weights to downstream tasks e.g. Weighted Least Squares
...
...
...
```





