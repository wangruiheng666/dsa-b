class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
    def __call__(self, left, right = None):
        self.left = left
        self.right = right
        return self
    
def qianxu(node):
    if node:
        print(node.val,end='')
        qianxu(node.left)
        qianxu(node.right)

def zhongxu(node):
    if node:
        zhongxu(node.left)
        print(node.val,end='')
        zhongxu(node.right)

for ss in range(int(input())):
    for ii in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz':
        exec(f'{ii} = Node("{ii}")')
    yy = input()
    root = eval(yy.replace('*','None'))
    qianxu(root)
    print()
    zhongxu(root)
    print()