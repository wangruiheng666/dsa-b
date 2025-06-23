from typing import Optional

def main():
    n,m = map(int,input().split())
    if n==m==0:
        raise EOFError
    # A:0,B:1,...
    adj = [[] for _ in range(n)]
    in_deg = [0 for _ in range(n)]
    determined:Optional[int] = None
    inconsistency:Optional[int] = None
    seq = [-1 for _ in range(n)] # 若该点已经排序了，在拓扑排序的位置
    sorted_list = []
    def has_cycle_in_unsorted():
        visited = set()
        res_stack = set()
        def dfs(node):
            visited.add(node)
            res_stack.add(node)
            for neighbor in adj[node]:
                if seq[neighbor] != -1:
                    continue
                if neighbor in res_stack:
                    return False
                if neighbor not in visited:
                    if not dfs(neighbor):
                        return False
            res_stack.remove(node)
            return True
        for i in range(n):
            if seq[i] == -1 and i not in visited:
                if not dfs(i):
                    return True
        return False
    for i in range(m):
        u, _, v = input()
        if inconsistency or determined:
            continue
        u_i = ord(u)-65
        v_i = ord(v)-65
        # 分情况讨论。
        # 1.uv都还没排序
        if seq[u_i] == -1 and seq[v_i] == -1:
            adj[u_i].append(v_i)
            in_deg[v_i] += 1
            if in_deg.count(0) == 0 or has_cycle_in_unsorted():
                inconsistency = i+1
                continue
            while in_deg.count(0) == 1:
                node = in_deg.index(0)
                in_deg[node] = -1
                sorted_list.append(node)
                if len(sorted_list) == n and not determined:
                    determined = i+1
                seq[node] = len(sorted_list) -1
                for neighbor in adj[node]:
                    in_deg[neighbor] -= 1
        # 2.uv都排序了
        elif seq[u_i] != -1 and seq[v_i] != -1:
            # u<v但u在v后，则矛盾。
            if seq[u_i] > seq[v_i]:
                inconsistency = i+1
        # 3.u没排序，v排序了，矛盾。
        elif seq[u_i] == -1 and seq[v_i] != -1:
            inconsistency = i+1
        # 3.v没排序，u排序了，无事发生。
    if inconsistency:
        print(f'Inconsistency found after {inconsistency} relations.')
    elif determined:
        print(f'Sorted sequence determined after {determined} relations: {"".join(chr(65+i) for i in sorted_list)}.')
    else:
        print('Sorted sequence cannot be determined.')

while 1:
    try:
        main()
    except EOFError:
        break
    