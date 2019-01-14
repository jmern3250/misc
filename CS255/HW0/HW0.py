import sys 

def strxor(a, b):     # xor two strings (trims the longer input)
    return "".join([chr(ord(x) ^ ord(y)) for (x, y) in zip(a, b)])

with open('DATA', 'r') as f: 
    data = f.readlines()

for i in range(10):
    for j in range(9 - i):
        delta = strxor(data[i][:-1].decode('hex'), data[j+1][:-1].decode('hex'))
        print('String %i and string %i difference \n' % (i, j+1))
        print(delta)
        import pdb; pdb.set_trace()