
# Euclidean_algorithym


def Euclidean_algo(m, n):
    while m%n != 0:
        m, n = n, m%n
    else:
        return n


m = int(input('the bigger int is :'))
n = int(input('the smaller int is :'))
print(Euclidean_algo(m, n))

# m,n must be valued at the same time
