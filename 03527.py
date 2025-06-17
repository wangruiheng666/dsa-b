from typing import List
from collections import Counter

def hu(nums: List[int]) -> str:
    if len(nums) not in (2,5,8,11,14):
        return 'XIANGGONG'
    nums.sort()
    used_count = 0
    not_used_count = dict(Counter(nums))
    pair_used = False # 有没有用到对子
    def dfs():
        nonlocal used_count,pair_used
        if used_count == len(nums):
            return True

        for card,count in not_used_count.items():
            if count > 0:
                break
        else:
            raise 114514
        
        if not_used_count[card] >= 2 and not pair_used:
            pair_used = True
            not_used_count[card] -= 2
            used_count += 2
            if dfs():
                return True
            pair_used = False
            not_used_count[card] += 2
            used_count -= 2
        
        if not_used_count[card] >= 3:
            not_used_count[card] -= 3
            used_count += 3
            if dfs():
                return True
            not_used_count[card] += 3
            used_count -= 3

        if not_used_count.get(card,0) and not_used_count.get(card+1,0) and not_used_count.get(card+2,0):
            not_used_count[card] -= 1
            not_used_count[card+1] -= 1
            not_used_count[card+2] -= 1
            used_count += 3
            if dfs():
                return True
            not_used_count[card] += 1
            not_used_count[card+1] += 1
            not_used_count[card+2] += 1
            used_count -= 3
        
        return False
    return 'HU' if dfs() else 'BUHU'

while 1:
    s = list(map(int,input().split()))
    if s == [0]:
        break
    print(hu(s))
