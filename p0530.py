import random

codes = []
for i in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
    codes.append(compile(f'{i} = random.randint(0,1000)','<string>','exec'))

def equal(aa, bb):
    for _ in range(10):
        for i in codes:
            exec(i,globals())
        ra = eval(aa)
        rb = eval(bb)
        if ra != rb:
            return False
    return True

for _ in range(int(input())):
    if equal(input(),input()):
        print('YES')
    else:
        print('NO')