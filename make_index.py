import sklearn
import sklearn.neighbors
import numpy as np
import pprint
import timeit


def main():
  data = np.random.normal(size=(1000000, 256))

  before = timeit.default_timer()
  tree = sklearn.neighbors.KDTree(data, leaf_size=100)
  after = timeit.default_timer()
  print(f'training took {after - before}s')

  pprint.pprint([(a.shape, a.dtype) for a in tree.get_arrays()])

if __name__ == '__main__':
  main()
