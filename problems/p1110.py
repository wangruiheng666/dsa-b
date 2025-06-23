n = int(input())
# 先是北，再是东
visited = [[False for _ in range(-n, n+1)] for _ in range(n+1)]
visited[0][0] = True
count = 0
def dfs(steps_remaining, i, j): #i,j:现在的位置
    global count
    if steps_remaining == 0:
        count += 1
       #print(i,j)
        return
    if not visited[i+1][j]:
        visited[i+1][j] = True
        dfs(steps_remaining-1, i+1, j)
        visited[i+1][j] = False
    if not visited[i][j+1]:
        visited[i][j+1] = True
        dfs(steps_remaining-1, i, j+1)
        visited[i][j+1] = False
    if not visited[i][j-1]:
        visited[i][j-1] = True
        dfs(steps_remaining-1, i, j-1)
        visited[i][j-1] = False
    
dfs(n, 0, 0)
print(count)