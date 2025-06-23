class Node:
    def __init__(self, val, children):
        self.val = val
        self.children = children
alphabet = lambda x : isinstance(x,str) and x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
stack = []
s = input()
for i in s:
    if alphabet(i):
        stack.append(Node(i,[]))
    if i == '(':
        stack.append(i)
    if i == ',':
        pass
    if i == ')':
        children = []
        while stack[-1] != '(':
            children.append(stack.pop())
        stack.pop()
        stack[-1].children = children[::-1]
root = stack[-1]
if alphabet(root):
    root = Node(root,[])

def dfs(node):
    print(node.val,end='')
    for i in node.children:
        dfs(i)

dfs(root)
print()
def dffs(node):
    for i in node.children:
        dffs(i)
    print(node.val,end='')
dffs(root)