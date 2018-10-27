
#LinearSearch

def LinearSearch(N, a):
    n = len(N)
    for i in range(n):
        if N[i] == a:
            return True

    else:
        return False

N = [13, 16, 23, 45, 54, 58, 76, 91]
a = 76

ans = LinearSearch(N, a)
print(ans)
#a in N でもpython なら同じ結果得られるよ
#for ループにelseを書くと最初の要素を比較してreturnで抜けてしまうので外に書かないと書き要素は比較できない


#BinarySearch

def BinarySearch(M, b):

    M.sort()
    n = len(M)

    if n == 0:
        return False

    elif M[n//2] == b:
        return True
    
    elif M[n//2] < b:
        return BinarySearch(M[n//2 + 1:], b)

    elif M[n//2] > b:
        return BinarySearch(M[:n//2 - 1], b)

M = N
b = a
ans2 = BinarySearch(M, b)
print(ans2)


#　最後にreturn BinarySearchにするとこ注意



    
