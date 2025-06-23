from collections import deque,defaultdict
import itertools
def main():
    n = int(input())
    adj_mat = [list(map(int,input().split())) for _ in range(n)]
    dp = [defaultdict(lambda:float('inf')) for _ in range(1<<n)]
    dp[1<<0][0] = 0

    def generate_numbers(k):
        if k < 0 or k > n:
            return
        # 生成所有包含k个1的位置组合
        for positions in itertools.combinations(range(n), k):
            num = 0
            for pos in positions:
                num |= (1 << pos)  # 将对应位置设为1
            yield num,positions


    for i in range(1,n):
        for condition,nodes in generate_numbers(i):
            for node in nodes:
                for new_node in range(n):
                    if node == new_node:
                        continue
                    if condition & (1<<new_node):
                        continue
                    new_condition = condition | (1<<new_node)
                    new_time = dp[condition][node] + adj_mat[node][new_node]
                    if new_time < dp[new_condition][new_node]:
                        dp[new_condition][new_node] = new_time

    print(dp[(1<<n) -1][n-1])

main()