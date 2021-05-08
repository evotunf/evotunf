# genetic algorithm search of the one max optimization problem
import math
import io
from typing import List, Dict

import numpy as np
import pandas as pd
from numpy.random import randint, rand, seed


def gauss(mu, sigma, x):
    return math.exp(-pow((x - mu)/sigma, 2))

def gauss_ex(fsets, i, j, x):
    mu, sigma = fsets[i][j]
    return gauss(mu, sigma, x)

def impl(a, b):
    return min(1, 1-a+b)  # Lukaszewicz


def infer(fsets: np.ndarray, rules: np.ndarray, xx: np.ndarray):
    num, den = 0.0, 0.0
    for *_, b in rules:
        uy_center, _ = fsets[-1][b-1]
        cross = 1.0
        for *aa, b in rules:
            m = max([
                    min(gauss_ex(fsets, i, x-1, t),
                        impl(gauss_ex(fsets, i, a-1, t),
                             gauss_ex(fsets, -1, b-1, uy_center)))*bool(a)
                    for i, (x, a) in enumerate(zip(xx, aa))
                    for t in np.arange(0, 1.01, 0.05)],
                default=1.0)
            cross = min(cross, m)
        num += uy_center * cross
        den += cross
    #return max(range(len(fsets[-1])),
               #key=lambda j: gauss_ex(-1, j, num / den))+1
    if den == 0:
        return 0
    res = [gauss_ex(fsets, -1, j, num / den) for j in range(len(fsets[-1]))]
    return np.argmax(res) + 1


fset_lens = [2, 4, 2, 2, 4, 2, 2]
fsets = [[(float(j+1) / (n+1), 1.0 / (n+1)) for j in range(n)] for i, n in enumerate(fset_lens)]
#for i, n in enumerate(fset_lens):
    #for j in range(n):
     #   fsets[i, j][0] = float(j+1) / (n+1)
      #  fsets[i, j][1] = 1.0 / (n+1)

samples = [
    [1,1,1,1,1,2,2],
    [2,1,1,1,1,1,1],
    [1,2,1,1,1,1,1],
    [1,1,1,1,1,1,1],
    [1,3,2,2,1,1,1],
    [1,1,1,1,4,1,1],
    [1,4,1,1,1,1,2],
    [1,4,1,1,2,1,2],
    [1,4,1,1,3,1,2],
    [1,3,1,1,1,1,2],
    [1,3,1,1,2,1,2],
    [1,3,1,2,1,1,2],
    [1,3,1,2,2,1,2],
    [1,3,1,1,3,1,1],
    [1,3,1,2,3,1,2],
]

infer(fsets, samples, samples[0])
 
# objective function
def onemax(x, n_rules):
    #x = ''.join(map(str, x))
    #a, b = int(x[:10], 2), int(x[10:], 2)
    #return math.exp(-(math.pow((a-4)/16, 2) + math.pow(b/16, 2))/2)
    rules = np.array(x).reshape(n_rules, -1)
    return sum(infer(fsets, rules, np.array(r[:-1])) == r[-1] for r in samples) / len(samples)

def compute_scores(pop, n_pop, n_rules):
    return [onemax(pop[i], n_rules) for i in range(n_pop)]
 
# tournament selection
def selection(pop, scores, k=3):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] > scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]
 
# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1)-2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]
 
# mutation operator
def mutation(rules, n_rules, r_mut):
    for i, bound in zip(range(len(rules)), np.tile(fset_lens, n_rules)):
        # check for a mutation
        if rand() < r_mut:
            # flip the bit
            rules[i] += 1
            rules[i] %= bound
 
# genetic algorithm
def genetic_algorithm(objective, n_rules, n_iter, n_pop, r_cross, r_mut):
    # initial population of random bitstring
    pop = [randint(0, np.tile(fset_lens, n_rules)).tolist() for _ in range(n_pop)]
    # keep track of best solution
    best, best_eval = pop[0], objective(pop[0], n_rules)
    # enumerate generations
    for gen in range(n_iter):
        # evaluate all candidates in the population
        #scores = [objective(c, n_rules) for c in pop]
        scores = compute_scores(pop, n_pop, n_rules)
        # check for new best solution
        for i in range(n_pop):
            if scores[i] > best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
        print(f'Min score: {min(scores):.3f} Max score: {max(scores):.3f} Avg score: {sum(scores)/len(scores):.3f}')
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, n_rules, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
    return [best, best_eval]
 
seed(42)

# define the total iterations
n_iter = 100
# bits
n_rules = 15
# define the population size
n_pop = 10
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / float(n_rules * 6)
# perform the genetic algorithm search
best, score = genetic_algorithm(onemax, n_rules, n_iter, n_pop, r_cross, r_mut)
print('Done!')
print('f(%s) = %f' % (best, score))