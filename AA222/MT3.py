import copy 
import numpy as np
import matplotlib.pyplot as plt 


#### P1 ####
def f(x, a=1., b=5.):
	return (a - x[0])**2 + b*(x[1] - x[0]**2)**2


# #### P2 ####
# X = np.arange(-2., 2., 0.01)
# m = X.shape[0]
# Z = np.zeros([m,m])

# for i in range(m):
# 	for j in range(m):
# 		Z[i,j] = f((X[j], X[i]))

# plt.figure()
# plt.contour(X,X,Z,levels=[0.5, 1, 5, 10, 20, 30, 50, 75, 100, 150])
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.show()

#### P3 ####

class Tree(object):
    def __init__(self):
        self.child = {}
        self.type = None
        self.value = None

    def set_type(self, type_str, const=None):
    	self.type = type_str
    	if type_str == 'const':
    		if const is None: 
    			self.value = np.random.randint(10)
    		else:
    			self.value = const
    	elif type_str =='x1' or type_str =='x2':
    		return
    	else: 
    		self.child[0] = Tree()
    		self.child[1] = Tree()

    def call(self, x1, x2):
    	if self.type == 'add':
    		return self.child[0].call(x1,x2) + self.child[1].call(x1,x2)
    	elif self.type == 'sub':
    		return self.child[0].call(x1,x2) - self.child[1].call(x1,x2)
    	elif self.type == 'mul':
    		return self.child[0].call(x1,x2) * self.child[1].call(x1,x2)
    	elif self.type == 'div':
    		return self.child[0].call(x1,x2) / self.child[1].call(x1,x2)
    	elif self.type == 'const':
    		return self.value
    	elif self.type == 'x1':
    		return x1
    	elif self.type =='x2':
    		return x2
    	else: 
    		return 

# # Example: (x1 + x2)*3
# tree = Tree()
# tree.set_type('mul')
# tree.child[0].set_type('const', 3.)
# tree.child[1].set_type('add')
# tree.child[1].child[0].set_type('x1')
# tree.child[1].child[1].set_type('x2')
# print(tree.call(1,2))

#### P4 ####

def update_children(tree, terminal=False):
	updated = 0 
	type_list = ['add', 'sub', 'mul', 'div', 'const', 'x1', 'x2']
	if terminal: 
		type_list = ['const', 'x1', 'x2']
	if tree.type is not None: 
		if not tree.child: 
			return updated  
		for child in tree.child.values():
			updated += update_children(child, terminal)
		return updated
	else:
		type_str = np.random.choice(type_list)
		tree.set_type(type_str)
		return 1 

def gen_random():
	
	done = False
	# Randomly initialize root
	tree = Tree()
	done = False 
	terminal = False
	steps = 0
	while not done: 
		updated = update_children(tree, terminal)
		done = updated == 0
		steps += 1
		terminal = steps >= 10
	return tree 
		
#### P5 ####

def gen_str(tree):
	string = []
	if tree.type == 'const':
		return str(tree.value)
	elif tree.type == 'x1' or tree.type == 'x2':
		return tree.type
	elif tree.type == 'add': 
		return '(' + gen_str(tree.child[0]) + '+' +  gen_str(tree.child[1]) + ')'
	elif tree.type == 'sub': 
		return '(' + gen_str(tree.child[0]) + str('-') +  gen_str(tree.child[1]) + ')'
	elif tree.type == 'mul': 
		return '(' + gen_str(tree.child[0]) + str('*') +  gen_str(tree.child[1]) + ')'
	elif tree.type == 'div': 
		return '(' + gen_str(tree.child[0]) + str('/') +  gen_str(tree.child[1]) + ')'

#### P6 ####

np.random.seed(0)
for _ in range(10):
	tree = gen_random()
	print()
	print(gen_str(tree))