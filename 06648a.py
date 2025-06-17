import heapq

def merge(a, b):
    heap = []
    n = len(a)
    for i in range(n):
        heapq.heappush(heap, (a[i] + b[0], i, 0))
    
    res = []
    for _ in range(n):
        val, i, j = heapq.heappop(heap)
        res.append(val)
        if j + 1 < len(b):
            next_val = a[i] + b[j+1]
            heapq.heappush(heap, (next_val, i, j+1))
    return res

t = int(input())
for _ in range(t):
    m , n = map(int,input().split())
    nums=[]
    for __ in range(m):
        nums.append(sorted(map(int,input().split())))
    sums = nums[0].copy()
    for i in range(1, m):
        sums = merge(sums, nums[i])
    
    print(' '.join(map(str, sums[:n])))