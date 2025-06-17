'''描述
求一个用字符矩阵表示的城堡中的房间个数和最大房间的面积。

输入
第一行是两个整数r和c(1 <=r,c <=100)表示字符矩阵共有r行c列。接下来的r行就是表示城堡的字符矩阵。'#'表示墙壁，'.'表示一块空地。左右或上下连在一起的空地构成房间。房间的面积就是其包含的字符'.'的数目。数据保证城堡最外围的一圈都是墙壁。
输出
输出城堡的房间数目和最大房间的面积
样例输入
6 12
############
#...#.#.#..#
#####......#
#..#########
#..#..#....#
############
样例输出
5
10'''
import heapq
from collections import deque
def bfs(start, r, c, grid, visited):
    queue = deque([start])
    area = 0
    while queue:
        x, y = queue.popleft()
        if visited[x][y]:
            continue
        visited[x][y] = True
        area += 1
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < r and 0 <= ny < c and not visited[nx][ny] and grid[nx][ny] == '.':
                queue.append((nx, ny))
    return area
def main():
    r, c = map(int, input().split())
    grid = [input().strip() for _ in range(r)]
    visited = [[False] * c for _ in range(r)]
    room_count = 0
    max_area = 0
    for i in range(r):
        for j in range(c):
            if grid[i][j] == '.' and not visited[i][j]:
                room_count += 1
                max_area = max(max_area, bfs((i, j), r, c, grid, visited))
    print(room_count)
    print(max_area)
if __name__ == "__main__":
    main()

# The code above defines a function to count the number of rooms and the maximum area of rooms in a castle represented by a character matrix.
# It uses a breadth-first search (BFS) algorithm to explore the rooms and calculate their areas.

# can you write a similar function using depth-first search (DFS) instead of BFS?

def dfs(start, r, c, grid, visited):
    stack = [start]
    area = 0
    while stack:
        x, y = stack.pop()
        if visited[x][y]:
            continue
        visited[x][y] = True
        area += 1
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < r and 0 <= ny < c and not visited[nx][ny] and grid[nx][ny] == '.':
                stack.append((nx, ny))
    return area
def main_dfs():
    r, c = map(int, input().split())
    grid = [input().strip() for _ in range(r)]
    visited = [[False] * c for _ in range(r)]
    room_count = 0
    max_area = 0
    for i in range(r):
        for j in range(c):
            if grid[i][j] == '.' and not visited[i][j]:
                room_count += 1
                max_area = max(max_area, dfs((i, j), r, c, grid, visited))
    print(room_count)
    print(max_area)
if __name__ == "__main__":
    main_dfs()

# can you write a similar function using disjoint set union (DSU) instead of DFS or BFS?

def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]
def union(parent, rank, x, y):
    rootX = find(parent, x)
    rootY = find(parent, y)
    if rootX != rootY:
        if rank[rootX] > rank[rootY]:
            parent[rootY] = rootX
        elif rank[rootX] < rank[rootY]:
            parent[rootX] = rootY
        else:
            parent[rootY] = rootX
            rank[rootX] += 1
def main_dsu():
    r, c = map(int, input().split())
    grid = [input().strip() for _ in range(r)]
    
    parent = {}
    rank = {}
    
    for i in range(r):
        for j in range(c):
            if grid[i][j] == '.':
                idx = i * c + j
                parent[idx] = idx
                rank[idx] = 0
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for i in range(r):
        for j in range(c):
            if grid[i][j] == '.':
                idx = i * c + j
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < r and 0 <= nj < c and grid[ni][nj] == '.':
                        nidx = ni * c + nj
                        union(parent, rank, idx, nidx)
    
    room_count = len(set(find(parent, x) for x in parent))
    max_area = max(sum(1 for x in parent if find(parent, x) == root) for root in set(find(parent, x) for x in parent))
    
    print(room_count)
    print(max_area)
if __name__ == "__main__":
    main_dsu()

# any other methods to solve this problem?
# Another method to solve this problem is using a flood fill algorithm, which is similar to DFS but can be implemented iteratively or recursively.

def flood_fill(start, r, c, grid, visited):
    stack = [start]
    area = 0
    while stack:
        x, y = stack.pop()
        if visited[x][y]:
            continue
        visited[x][y] = True
        area += 1
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < r and 0 <= ny < c and not visited[nx][ny] and grid[nx][ny] == '.':
                stack.append((nx, ny))
    return area
def main_flood_fill():
    r, c = map(int, input().split())
    grid = [input().strip() for _ in range(r)]
    visited = [[False] * c for _ in range(r)]
    room_count = 0
    max_area = 0
    for i in range(r):
        for j in range(c):
            if grid[i][j] == '.' and not visited[i][j]:
                room_count += 1
                max_area = max(max_area, flood_fill((i, j), r, c, grid, visited))
    print(room_count)
    print(max_area)
if __name__ == "__main__":
    main_flood_fill()

# The code above defines a function to count the number of rooms and the maximum area of rooms in a castle represented by a character matrix using a flood fill algorithm.
# It uses an iterative approach to explore the rooms and calculate their areas.
# any other methods to solve this problem?
# Another method to solve this problem is using a flood fill algorithm, which is similar to DFS but can be implemented iteratively or recursively.

def flood_fill_recursive(x, y, r, c, grid, visited):
    if x < 0 or x >= r or y < 0 or y >= c or visited[x][y] or grid[x][y] == '#':
        return 0
    visited[x][y] = True
    area = 1  # Count the current cell
    # Explore all four directions
    area += flood_fill_recursive(x - 1, y, r, c, grid, visited)
    area += flood_fill_recursive(x + 1, y, r, c, grid, visited)
    area += flood_fill_recursive(x, y - 1, r, c, grid, visited)
    area += flood_fill_recursive(x, y + 1, r, c, grid, visited)
    return area
def main_flood_fill_recursive():
    r, c = map(int, input().split())
    grid = [input().strip() for _ in range(r)]
    visited = [[False] * c for _ in range(r)]
    room_count = 0
    max_area = 0
    for i in range(r):
        for j in range(c):
            if grid[i][j] == '.' and not visited[i][j]:
                room_count += 1
                area = flood_fill_recursive(i, j, r, c, grid, visited)
                max_area = max(max_area, area)
    print(room_count)
    print(max_area)
if __name__ == "__main__":
    main_flood_fill_recursive()
# The code above defines a function to count the number of rooms and the maximum area of rooms in a castle represented by a character matrix using a recursive flood fill algorithm.
# It uses recursion to explore the rooms and calculate their areas.
# Compare the complexity of the different methods used to solve this problem.
# The complexity of the different methods used to solve this problem can be analyzed as follows:
# 1. **BFS and DFS**:
#    - Both BFS and DFS have a time complexity of O(r * c) in the worst case, where r is the number of rows and c is the number of columns in the grid. This is because each cell is visited at most once.
#    - The space complexity is also O(r * c) for the visited array and the queue (for BFS) or stack (for DFS).
# 2. **Disjoint Set Union (DSU)**:
#    - The DSU method has a time complexity of O((r * c) * α(r * c)), where α is the inverse Ackermann function, which grows very slowly. In practice, this is nearly linear.
#    - The space complexity is O(r * c) for the parent and rank arrays.
# 3. **Flood Fill (Iterative)**:
#    - The iterative flood fill has a time complexity of O(r * c) similar to BFS and DFS, as each cell is visited at most once.
#    - The space complexity is O(r * c) for the visited array and the stack.
# 4. **Flood Fill (Recursive)**:
#    - The recursive flood fill also has a time complexity of O(r * c) since each cell is visited at most once.
#    - The space complexity is O(r * c) for the visited array and the recursion stack, which can go as deep as r * c in the worst case.
# In summary, all methods have a time complexity of O(r * c), but the DSU method has a slightly higher overhead due to the union-find operations. The space complexity is also similar across all methods, primarily due to the visited array. The choice of method may depend on specific constraints or preferences for iterative vs. recursive approaches.
