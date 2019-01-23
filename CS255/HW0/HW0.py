import sys 

def strxor(a, b):     # xor two strings (trims the longer input)
    return "".join([chr(ord(x) ^ ord(y)) for (x, y) in zip(a, b)])

with open('DATA', 'r') as f: 
    data = f.readlines()

def potential_spaces(delta):
    idxs = set()
    for i in range(len(delta)):
        if delta[i].isalpha():
            idxs.add(i)
    return idxs
def make_guess(c_string, data, space_idcs):
    m = len(c_string.decode('hex'))
    guess = ['-']*m
    for i in range(10):
        delta = strxor(data[i][:-1].decode('hex'), c_string.decode('hex'))
        for idx in space_idcs[i]:
            if idx >= m:
                break
            guess[idx] = strxor(delta[idx], ' ')    
    guess_string = "".join(x for x in guess)
    print(guess_string)
    return guess 

def crib_drag(string_0, string_1, crib_string, guess):
    delta = strxor(string_0.decode('hex'), string_1.decode('hex'))
    m = len(delta)
    n = len(crib_string)
    for i in range(m - n):
        test_delta = strxor(delta[i:i+n], crib_string) 
        # if test_delta.replace(' ','').replace(':','').replace('\x7f','').isalpha():
        if True:
            print(test_delta)
            accept = input('Accept?')
            if accept:
                guess[i:i+n] = [x for x in test_delta]
                guess_string = "".join(x for x in guess)
                print(guess_string)


space_idcs = {}
for i in range(11):
    idcs = None
    if i == 10:
        i_data = data[i].decode('hex')
    else:
        i_data = data[i][:-1].decode('hex')
    for j in range(11):
        if i != j:
            if j == 10:
                j_data = data[j].decode('hex')
            else:
                j_data = data[j][:-1].decode('hex')
            delta = strxor(i_data, j_data)
            idxs = potential_spaces(delta)
            if idcs is None: 
                idcs = idxs 
            else:
                idcs = idcs.intersection(idxs)
    space_idcs[i] = idcs

guess0 = make_guess(data[0][:-1], data, space_idcs)
guess1 = make_guess(data[1][:-1], data, space_idcs)
guess2 = make_guess(data[2][:-1], data, space_idcs)
guess3 = make_guess(data[3][:-1], data, space_idcs)
guess4 = make_guess(data[4][:-1], data, space_idcs)
guess5 = make_guess(data[5][:-1], data, space_idcs)
guess6 = make_guess(data[6][:-1], data, space_idcs)
guess7 = make_guess(data[7][:-1], data, space_idcs)
make_guess(data[8][:-1], data, space_idcs)
make_guess(data[9][:-1], data, space_idcs)

guess = make_guess(data[10], data, space_idcs)
# crib_drag(data[10], data[0][:-1], ' number ', guess)
# crib_drag(data[10], data[0][:-1], ' the ', guess)

# # crib_drag(data[0][:-1], data[10], ' message ', guess0)
# # crib_drag(data[0][:-1], data[10], ' than ', guess0)
# # crib_drag(data[0][:-1], data[10], ' is: ', guess0)
# # crib_drag(data[0][:-1], data[10], ' cipher ', guess0)
# # crib_drag(data[0][:-1], data[10], 'The ', guess0)

# crib_drag(data[10], data[0][:-1], ' computer', guess)
# crib_drag(data[10], data[0][:-1], ' fast ', guess)

# # crib_drag(data[7][:-1], data[10], ' message ', guess7)
# # crib_drag(data[7][:-1], data[10], ' than ', guess7)
# # crib_drag(data[7][:-1], data[10], ' is: ', guess7)
# # crib_drag(data[7][:-1], data[10], ' cipher ', guess7)
# # crib_drag(data[7][:-1], data[10], 'The ', guess7)
# # crib_drag(data[7][:-1], data[10], ' central ', guess7)

# crib_drag(data[10], data[7][:-1], ' the point ', guess)
# crib_drag(data[10], data[7][:-1], ' where ', guess)

# # crib_drag(data[0][:-1], data[10], 'The ', guess0)
# # crib_drag(data[1][:-1], data[10], 'The ', guess1)
# # crib_drag(data[5][:-1], data[10], 'The secret', guess5)
# crib_drag(data[10], data[5][:-1], 'There ', guess)
# # crib_drag(data[0][:-1], data[10], 'The secret message is', guess0)
# crib_drag(data[0][:-1], data[10], ' using a stream cipher, never ', guess0)
crib_drag(data[1][:-1], data[10], 'The secret message is: When using a stream cipher, never use the', guess1)

# import pdb; pdb.set_trace()
guess_string = 'The secret message is: When using a stream cipher, never use the key more than once'

# guess_string = "".join(x for x in guess)
# print(guess_string)