n = 20
m = 30
import random
print(n,m)
for _ in range(m):
    print(random.randint(0,n-1),random.randint(0,n-1))
k = 30
print(k)
for _ in range(k):
    x = list(range(n))
    random.shuffle(x)
    print(*x)