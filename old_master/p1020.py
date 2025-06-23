n,m = map(int,input().split())
adj = [[] for _ in range(n)]
for _ in range(m):
    u,v = map(int,input().split())
    adj[u].append(v)
    adj[v].append(u)

visited = set()
def dfs(node):
    if node in visited:
        return
    print(node, end=' ')
    visited.add(node)
    for neighbor in adj[node]:
        if neighbor not in visited:
            dfs(neighbor)

for i in range(n):
    if i not in visited:
        dfs(i)