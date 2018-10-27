
#ヒープをリストで実装

#ヒープはデータ構造 data[]と最後の要素番号 lastを持つとする


class Heapclass:
    def __init__(self, )






#要素の追加バージョン

def insert(heap, object):
    heap.last = heap.last + 1
    heap.data[heap.last] = object
    i = heap.last
    while i > 0:
        if heap.data[i] > heap.data[(i-1)//2]:

