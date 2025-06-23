from collections import deque
for k in range(int(input())):
    def valid(l):
        if 0<=l[0]<m and 0<=l[1]<n:
            return True
        return False

    queue=deque()
    m,n=map(int,input().split())
    board=[]
    for _ in range(m):
        board.append(list(map(int,input().split())))
    x,y = map(int,input().split())
    head=[x-1,y-1]
    for _ in range(int(input())):
        x,y = map(int,input().split())
        queue.append([x-1,y-1])
    ans=0
    while queue:
        cur=queue.popleft()
        if cur==head:
            ans=1
            break
        if board[cur[0]][cur[1]]!=1001:
            if valid([cur[0]+1,cur[1]]):
                if board[cur[0]][cur[1]]>board[cur[0]+1][cur[1]]:
                    queue.append([cur[0]+1,cur[1]])
            if valid([cur[0],cur[1]+1]):
                if board[cur[0]][cur[1]]>board[cur[0]][cur[1]+1]:
                    queue.append([cur[0],cur[1]+1])
            if valid([cur[0]-1,cur[1]]):
                if board[cur[0]][cur[1]]>board[cur[0]-1][cur[1]]:
                    queue.append([cur[0]-1,cur[1]])
            if valid([cur[0],cur[1]-1]):
                if board[cur[0]][cur[1]]>board[cur[0]][cur[1]-1]:
                    queue.append([cur[0],cur[1]-1])
            board[cur[0]][cur[1]]=1001
    if ans:
        print("Yes")
    else:
        print("No")