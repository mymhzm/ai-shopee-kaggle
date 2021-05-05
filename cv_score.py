import numpy as np
def getMetric(col):
    def f1score(row):
        n = len(np.intersect1d(row.target, row[col]))
        return 2*n / (len(row.target) + len(row[col]))
    return f1score