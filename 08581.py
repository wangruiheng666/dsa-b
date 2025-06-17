from typing import Optional

s = input()
it = iter(s)

class Node:
    def __init__(self, val):
        self.val = val
        self.left = self.right = None

def dfs() -> Optional[Node]:
    val = next(it)
    if val == '.':
        return None
    node = Node(val)
    node.left = dfs()
    node.right = dfs()
    return node

root = dfs()

def zhongxu(node: Optional[Node]):
    if node:
        zhongxu(node.left)
        print(node.val,end='')
        zhongxu(node.right)

def houxu(node : Optional[Node]):
    if node:
        houxu(node.left)
        houxu(node.right)
        print(node.val,end='')

zhongxu(root)
print()
houxu(root)