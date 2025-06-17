from functools import lru_cache
import sys
sys.setrecursionlimit(2147483647)

str1 = input()
str2 = input()

i = 0
j = 0

#@lru_cache(maxsize=None)
def match() -> bool:
    global i,j
    if i>=len(str1):
        return i==len(str1) and j==len(str2)
    if j>len(str2):
        return False
    
    if str1[i] == '?':
        i += 1
        j += 1
        a = match()
        i -= 1
        j -= 1
        return a
    elif str1[i] == '*':
        # 如果*匹配了0个字符
        i += 1
        if match():
            i -= 1
            return True
        # 如果*匹配了1个字符
        j += 1
        if match():
            i -= 1
            j -= 1
            return True
        # 如果*匹配了多个字符
        i -= 1
        a = match()
        i += 1
        return a
    elif j<len(str2) and str1[i] == str2[j]:
        i += 1
        j += 1
        a = match()
        i -= 1
        j -= 1
        return a
    else:
        return False

s = 0
while s < len(str1) -1:
    if str1[s] == str1[s+1] == '*':
        str1 = str1[:s] +str1[s+1:]
    s += 1
 
print('matched' if match() else 'not matched')


