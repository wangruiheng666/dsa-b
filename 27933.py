stack = []
removed_cnt = 0
res = 0
n = int(input())
for _ in range(2*n):
    cmd = input()
    if cmd[0] == 'a':
        stack.append(int(cmd.split()[-1]))
    if cmd[0] == 'r':
        removed_cnt += 1
        if stack[-1] != removed_cnt:
            stack.sort(reverse=True)
            stack.pop()
            res += 1
        else:
            stack.pop()
print(res)