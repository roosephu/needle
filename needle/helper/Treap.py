import numpy as np

cdf = None

class Node:
    def __init__(self, lc=null, rc=null, key=0., value=None, weight=0.):
        self.lc = lc
        self.rc = rc
        self.aux = np.random.rand()
        self.sum = weight
        self.value = value
        self.size = 1.
        self.key = key

    def update(self):
        self.sum = self.lc.sum + self.rc.sum + self.weight
        self.size = self.lc.size + self.rc.size + 1.
        return self

null = Node(null, null)
null.size = 0
null.aux = 1.

class Treap:
    def __init__(self, cdf=None):
        self.root = null
        cdf = cdf

    @staticmethod
    def split(root, key):
        if root == null:
            return null, null
        if key <= root.lc.key:
            root.rc, rc = split(root.rc, key)
            return root.update(), rc
        else:
            lc, root.lc = split(root.lc, key)
            return lc, root.update()

    @staticmethod
    def merge(lc, rc):
        if lc.aux <= rc.aux:
            lc.rc = merge(lc.rc, rc)
            return update(lc)
        else:
            rc.lc = merge(lc, rc.lc)
            return update(rc)

    def add(self, value, priority):
        lc, rc = split(self.root, priority)
        node = Node(null, null, priority, value, 1.)
        self.root = merge(lc, merge(node, rc))

    def sample(self):
        if self.cdf == None:
            weights = self.root.sum
        else:
            weights = self.cdf[self.root.size]
        self.root.sample(weights * np.random.rand())