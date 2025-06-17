import heapq
from typing import Dict,List,Tuple

def main():
    p = int(input())
    num_to_name:List[str] = []
    name_to_num:Dict[str,int] = {}
    adj:Dict[int,Dict[int,int]] = {}
    for i in range(p):
        name = input()
        adj[i] = {}
        num_to_name.append(name)
        name_to_num[name] = i
    q = int(input())
    for _ in range(q):
        u,v,d = input().split()
        d = int(d)
        u = name_to_num[u]
        v = name_to_num[v]
        if v not in adj[u] or d < adj[u][v]:
            adj[u][v] = d
            adj[v][u] = d
    r = int(input())
    for _ in range(r):
        s,e = input().split()
        if s == e:
            print(s)
            continue
        s = name_to_num[s]
        e = name_to_num[e]

        heap:List[Tuple[int,int,Tuple[int]]] = [(0,s,(s,))]
        visited = [False for _ in range(p)]
        visited[s] = True
        res = []
        # 顶点，总花费，经过的顶点列表
        while heap:
            cost, node,  path = heapq.heappop(heap)
            
            visited[node] = True
            if node == e:
                res = path
                break
            for neighbor,fee in adj[node].items():
                if not visited[neighbor]:
                    new_path = path + (neighbor,)
                    heapq.heappush(heap, (cost+fee, neighbor, new_path))
        
        for i in range(len(res)-1):
            print(num_to_name[res[i]],end='->')
            print(f'({adj[res[i]][res[i+1]]})',end='->')
        print(num_to_name[res[-1]])

main()