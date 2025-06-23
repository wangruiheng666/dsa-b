n,m = map(int,input().split())
parent = list(range(n))
rank = [1]*n
min_cost = list(map(int,input().split()))
def find(node):
    while parent[node] != node:
        node = parent[node]
    return node
def union(x,y):
    px,py = find(x),find(y)
    if px==py:
        return
    if rank[px] < rank[py]:
        parent[px] = py
        min_cost[py] = min(min_cost[px], min_cost[py])
    elif rank[px] > rank[py]:
        parent[py] = px
        min_cost[px] = min(min_cost[px], min_cost[py])
    else:
        parent[py] = px
        rank[px] += 1
        min_cost[px] = min(min_cost[px], min_cost[py])
for _ in range(m):
    x,y = map(int,input().split())
    union(x-1,y-1)
res = 0
for i in range(n):
    if i == parent[i]:
        res += min_cost[i]
print(res)