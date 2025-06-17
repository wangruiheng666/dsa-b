import heapq
from collections import defaultdict

vertexs = set()
adj = defaultdict(dict)
min_times = defaultdict(lambda: 1145141919810)
# adj = {start:{end:time}}

x1,y1,x2,y2 = map(int,input().split())
vertexs.add((x1,y1))
vertexs.add((x2,y2))

def distance(xa,ya,xb,yb):
    return ((xa-xb)**2 + (ya-yb)**2) **0.5

while 1:
    try:
        subway = list(map(int,input().split()))[:-2]
        prev_station = None
        for i in range(0, len(subway), 2):
            x,y = subway[i:i+2]
            vertexs.add((x,y))
            if prev_station:
                # 40km/h = 40000m/60min = (2000/3) m/min
                adj[prev_station][(x,y)] = distance(x,y,*prev_station)/(2000/3)
                adj[(x,y)][prev_station] = distance(x,y,*prev_station)/(2000/3)
            prev_station = (x,y)
    except EOFError:
        break

for i in vertexs:
    for j in vertexs:
        if i == j:
            continue
        if j not in adj[i]:
            # 10km/h = 10000m/60min = (500/3) m/min
            adj[i][j] = distance(*i,*j)/(500/3)

heap = [(0,(x1,y1))] # [(time,(x,y))]

while heap:
    time,(x,y) = heapq.heappop(heap)
    if x==x2 and y==y2:
        print(round(time))
        break
    if time > min_times[(x,y)]:
        continue
    min_times[(x,y)] = time
    for xn,yn in adj[(x,y)]:
        new_time = time + adj[(x,y)][(xn,yn)]
        if new_time >= min_times[(xn,yn)]:
            continue
        min_times[(xn,yn)] = new_time
        heapq.heappush(heap, (new_time,(xn,yn)))
