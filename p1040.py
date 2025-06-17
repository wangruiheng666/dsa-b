n,m = map(int, input().split())
adj = [[] for _ in range(n)]
for _ in range(m):
    u,v = map(int, input().split())
    adj[u].append(v)
    adj[v].append(u)

visited = set()
def dfs(node):
    visited.add(node)
    for new_node in adj[node]:
        if new_node not in visited:
            dfs(new_node)
dfs(0)
print(f'connected:{'yes' if len(visited) == n else 'no'}')
visited = set()

def new_dfs(node, parent):
    visited.add(node)
    for new_node in adj[node]:
        if new_node not in visited:
            if not new_dfs(new_node, node):
                return False
        elif new_node == parent:
            pass
        else:
            return False
    return True

def have_loop():
    for i in range(n):
        if i not in visited:
            if not new_dfs(i, None):
                return True
    return False

print(f'loop:{'yes' if have_loop() else 'no'}')