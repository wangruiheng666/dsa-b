import heapq

t = int(input())
for _ in range(t):
    m,n = map(int,input().split()) # m个，每个n个数
    nums = [sorted(map(int,input().split())) for _ in range(m)]
    curr_sums = nums[0]
    for i in range(1, m):
        prev_sums = curr_sums.copy()
        curr_sums.clear()
        curr = nums[i]
        heap = [(prev_sums[0] + curr[0], 0, 0)]
        heapq.heapify(heap)
        visited = {(0,0)}
        for j in range(n):
            s, i, j = heapq.heappop(heap)
            curr_sums.append(s)
            if i != n-1 and (i+1,j) not in visited:
                visited.add((i+1,j))
                heapq.heappush(heap, (prev_sums[i+1] + curr[j], i+1, j))
            if j != n-1 and (i,j+1) not in visited:
                visited.add((i,j+1))
                heapq.heappush(heap, (prev_sums[i] + curr[j+1], i, j+1))
    print(' '.join(map(str,curr_sums)))
    
