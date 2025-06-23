n,m = map(int,input().split())
adj = [set() for _ in range(n)]
for _ in range(m):
    u,v = map(int,input().split())
    adj[u].add(v)
    adj[v].add(u)

def is_dfs_path(path:list) -> bool:
    cur_index = 0
    visited = [False for _ in range(n)]
    isnt_dfs_path = False
    def dfs():
        nonlocal cur_index,isnt_dfs_path
        node = path[cur_index]
        if visited[node]:
            return
        visited[node] = True
        while 1:
            neighbors = [n for n in adj[node] if not visited[n]]
            if not neighbors:
                break
            cur_index += 1
            if  path[cur_index] not in neighbors:
                isnt_dfs_path = True
                return
            dfs()
    while cur_index < n:
        dfs()
        if isnt_dfs_path:
            return False
        cur_index += 1
    return not isnt_dfs_path



k = int(input())
for _ in range(k):
    path = list(map(int,input().split()))
    if len(path) != n or set(path) != set(range(n)):
        print('NO')
    elif is_dfs_path(path):
        print('YES')
    else:
        print('NO')