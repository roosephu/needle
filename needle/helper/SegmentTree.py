import numpy as np


class Node:
    def __init__(self, value=None, lc=None, rc=None):
        self.lc = lc
        self.rc = rc
        self.sum = 0.
        self.value = value

    def update(self, l, r, pos, value, weight):
        if l == pos and r == pos + 1:
            self.sum = weight
            self.value = value
            return
        m = (l + r) // 2
        if pos < m:
            self.lc.update(l, m, pos, value, weight)
        else:
            self.rc.update(m, r, pos, value, weight)

        self.sum = self.lc.sum + self.rc.sum

    def sample(self, l, r, weight):
        if l + 1 == r:
            return self.value
        m = (l + r) // 2
        if weight <= self.lc.sum:
            return self.lc.sample(l, m, weight)
        else:
            return self.rc.sample(m, r, weight - self.lc.sum)

    def find(self, l, r, index):
        if l + 1 == r:
            return self.value
        m = (l + r) // 2
        if index < m:
            return self.lc.find(l, m, index)
        else:
            return self.rc.find(m, r, index)

null = Node()


class SegmentTree:
    def __init__(self, size): # can only sample by weight
        self.size = size
        self.root = self.build(0, size)

    def build(self, l, r):
        if l + 1 == r:
            return Node(None, null, null)
        m = (l + r) // 2
        lc, rc = null, null
        if l != m:
            lc = self.build(l, m)
        if m != r:
            rc = self.build(m, r)
        return Node(None, lc, rc)

    def update(self, pos, value, weight):
        self.root.update(0, self.size, pos, value, weight)

    def sample(self, rand=None):
        if rand is None:
            rand = np.random.rand()
        return self.root.sample(0, self.size, rand * self.root.sum)

    def find(self, index):
        return self.root.find(0, self.size, index)


def main():
    n = 1000
    segment_tree = SegmentTree(n)
    x = [(0., None) for i in range(n)]

    for i in range(n):
        if np.random.rand() > 0.5:
            t = np.random.randint(0, n)
            v = np.random.rand()
            w = np.random.rand()
            segment_tree.update(t, v, w)
            x[t] = (w, v)
        else:
            s = 0
            for w, v in x:
                s += w

            r = np.random.rand()
            s *= r
            v1 = segment_tree.sample(r)
            for w, v in x:
                if w >= s:
                    print v1 == v
                    break
                else:
                    s -= w

if __name__ == "__main__":
    main()
