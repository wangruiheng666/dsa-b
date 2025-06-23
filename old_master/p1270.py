import heapq

n,m = map(int,input().split())
pokemons = {i:int(input()) for i in range(2,n+2)}
pokemons[0] = pokemons[1] = 0
adj = {i:[] for i in range(n+2)}

for i in range(m):
    start,end,time = map(int,input().split())
    adj[start].append((end, time))
    adj[end].append((start, time))

heap = [(0,0)] # time, vertex

min_times = [1145141919810 for _ in range(n+2)]
min_times[0] = 0

while heap:
    time, vertex = heapq.heappop(heap)
    if vertex == 1:
        print(time)
        break
    for new_vertex, road_time in adj[vertex]:
        pokemon_time = pokemons[new_vertex]
        new_time = time + road_time + pokemon_time
        if new_time < min_times[new_vertex]:
            min_times[new_vertex] = new_time
            heapq.heappush(heap, (new_time,new_vertex))
