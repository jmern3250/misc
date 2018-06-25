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
#   for j in range(m):
#       Z[i,j] = f((X[j], X[i]))

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
        self.n_child = 0
        self.id = None

    @property 
    def n_subtree(self):
        value = 1
        for child in self.child.values():
            value += child.n_subtree
        return value

    def set_type(self, type_str, const=None):
        self.type = type_str
        if type_str == 'const':
            if const is None: 
                self.value = np.random.randint(9) + 1.
            else:
                self.value = const
        elif type_str =='x1' or type_str =='x2':
            return
        else: 
            self.child[0] = Tree()
            self.child[1] = Tree()
            if self.id is None: 
                self.child[0].id = '0'
                self.child[1].id = '1'
            else:
                self.child[0].id = self.id + '0'
                self.child[1].id = self.id + '1'
            self.n_child = 2

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

def gen_random(depth=5):
    
    done = False
    tree = Tree()
    done = False 
    terminal = False
    steps = 0
    while not done: 
        updated = update_children(tree, terminal)
        done = updated == 0
        steps += 1
        terminal = steps >= depth
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

# np.random.seed(0)
# for _ in range(10):
#   tree = gen_random()
#   print()
#   print(gen_str(tree))

#### P7 #### 

# np.random.seed(2)
# tree = gen_random()
# print(gen_str(tree))
# X = np.arange(-2., 2., 0.01)
# m = X.shape[0]
# Z = np.zeros([m,m])

# for i in range(m):
#   for j in range(m):
#       Z[i,j] = tree.call(X[j], X[i])

# plt.figure()
# plt.contour(X,X,Z,50)
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.show()

#### P8 #### 

def L1(tree, f):
    X = np.arange(-2., 2., 0.2)
    m = X.shape[0]
    eta = 1./m**2
    loss = 0.0
    for i in range(m):
        for j in range(m):
            try: 
                t = tree.call(X[i], X[j])
                s = f((X[i], X[j]))
                err = np.abs(t - s)
                err = np.clip(err, 0, 1e4)
                err *= eta
            except ZeroDivisionError:
                err = 1e4*eta
            loss += err
    return loss

#### P9 ####
# np.random.seed(0)
# for _ in range(10):
#   tree = gen_random()
#   loss = L1(tree, f)
#   print(gen_str(tree))
#   print('L1 loss:', loss)
#   print()

#### P10 #### 

def mutate(tree, n_nodes=None):
    if n_nodes is None:
        n_nodes = tree.n_subtree
    logit = np.random.uniform(0.,1.)
    if logit <= 1./n_nodes:
        if tree.n_subtree >= 100:
            return gen_random(5), True
        else:
            return gen_random(0), True
    else: 
        n_nodes -= 1
        if not tree.child: 
            return tree, False
        else: 
            for i, child in tree.child.items():
                node, mutated = mutate(child, n_nodes)
                n_nodes -= 1
                if mutated:
                    tree.child[i] = node 
                    return tree, mutated 
            return node, False


# np.random.seed(1)
# tree = gen_random()
# print(gen_str(tree))
# tree, _ = mutate(tree)
# print()
# print(gen_str(tree))

#### P11 #### 
def sample_node(tree, n_nodes=None):
    if n_nodes is None:
        n_nodes = tree.n_subtree
    logit = np.random.uniform(0.,1.)
    if logit <= 1./n_nodes:
        return tree
    else: 
        n_nodes -= 1
        if not tree.child: 
            return None
        else: 
            for child in tree.child.values():
                node = sample_node(child)
                n_nodes -= 1
                if node is not None:
                    return node
            return node

def replace(tree, node, n_nodes=None):

    if n_nodes is None:
        n_nodes = tree.n_subtree
    logit = np.random.uniform(0.,1.)
    if logit <= 1./n_nodes:
        return node, True
    else: 
        n_nodes -= 1
        if not tree.child: 
            return tree, False
        else: 
            for i, child in tree.child.items():
                node, replaced = replace(child, node, n_nodes)
                n_nodes -= 1
                if replaced:
                    tree.child[i] = node 
                    return tree, replaced 
            return node, False

def crossover(parent_a, parent_b):
    node = sample_node(parent_b)
    tree, _ = replace(parent_a, node)
    return tree


# # np.random.seed(5)
# tree_a = gen_random()
# tree_b = gen_random()
# print('Parent A:', gen_str(tree_a))
# print()
# print('Parent B:', gen_str(tree_b))
# tree = crossover(tree_a, tree_b)

# print()
# print('Child:', gen_str(tree))

#### P12 ####

def eval_pop(population):
    ''' Generates list of scores using L1 evaluation function'''
    scores = []
    for member in population: 
        scores.append(L1(member, f))
    return scores


def selection(population, scores, n_survivors):
    ''' Implements elite sample selection '''
    idxs = []
    for i in range(n_survivors):
        idx = np.nanargmin(scores)
        scores[idx] = np.inf
        idxs.append(idx)
    survivors = [population[i] for i in idxs]
    return survivors

def gen_population(survivors, n_population, p_crossover=0.5,debug=False):
    ''' Applies crossover and mutation to generate new population'''
    n_survivors = len(survivors)
    if n_survivors >= n_population: 
        raise ValueError('Population size must exceed number of survivors!')
    new_pop = survivors 
    for i in range(n_population - n_survivors):
        if np.random.uniform() <= p_crossover:
            idxs = np.random.choice(np.arange(n_survivors, dtype=np.int64), size=2)
            parents = [copy.deepcopy(survivors[i]) for i in idxs]
            child = crossover(*parents)
        else:
            idx = np.random.choice(np.arange(n_survivors, dtype=np.int64), size=1)
            parent = copy.deepcopy(survivors[idx[0]])
            child, _ = mutate(parent)
        new_pop.append(child)
    return new_pop

def main(init_population, n_itrs=50, population_size=150, 
        n_survivors=25, p_crossover=0.25):
    ''' Runs the genetic algorithm ''' 
    mean_scores = []
    std_scores = []
    min_scores = []
    survivors = init_population
    for i in range(n_itrs):
        debug = False
        if i >= 6:
            debug = True 
        population= gen_population(survivors, 
                    n_population=population_size,
                    p_crossover=p_crossover,debug=debug)

        scores = eval_pop(population)
        # import pdb; pdb.set_trace()
        mean_scores.append(np.nanmean(scores))
        std_scores.append(np.nanstd(scores))
        min_scores.append(np.nanmin(scores))

        survivors = selection(population, scores, n_survivors)
        print('%r iterations of %r complete' % ((i+1), n_itrs))
        print('Current best score: %r' % np.nanmin(scores))

    elite_scores = eval_pop(survivors)
    data = {'elite_pop':survivors,
            'elite_scores':elite_scores,
            'mean_scores':mean_scores,
            'std_scores':std_scores,
            'min_scores':min_scores}
    return data

#### P13 #### 

np.random.seed(1)
init_population = []
for _ in range(20):
    init_population.append(gen_random(5))

print('Population Initialized')
results = main(init_population, n_itrs=100)

import pdb; pdb.set_trace()

# plt.figure()
# plt.plot(results['mean_scores'], '--')
# plt.plot(results['min_scores'], '-.')
# plt.yscale('log')
# plt.grid(True)
# plt.xlabel('Generation')
# plt.ylabel('L1 Loss')
# plt.title('Performance History')
# plt.legend(['Mean Score', 'Minimum Score'])
# plt.show()
