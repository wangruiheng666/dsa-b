import sys
input = map(int, sys.stdin.read().split())

def main():
    n = next(input)
    m = next(input)
    parent = list(range(n))
    rank = [1 for _ in range(n)]
    coke = [True for _ in range(n)]
    def find(node):
        while node != parent[node]:
            node = parent[node]
        return node
    def union(x, y):
        px = find(x)
        py = find(y)
        if px == py:
            print('Yes')
        else:
            print('No')
            coke[py] = False
            if rank[px] < rank[py]:
                parent[px] = py
            elif rank[px] > rank[py]:
                parent[py] = px
            else:
                parent[py] = px
                rank[py] += 1
            
    for _ in range(m):
        x = next(input)
        y = next(input)
        union(x-1,y-1)
    res = []
    for i in range(n):
        if coke[i]:
            res.append(i+1)
    res.sort()
    print(len(res))
    print(*res)

while 1:
    try:
        main()
    except StopIteration:
        break