import numpy as np
import functools
import Fitness_NN as fnn


def makeGen(l_gen):
    # This function make a string with (0,1) of length l_gen
    gen = ""
    list_bin = np.random.randint(2, size=l_gen)
    for count in range(l_gen):
        gen += str(list_bin[count])
    return (gen)

def genInitPop(indi, var, len_v):
    # This function create the first population
    tmp_pop = np.ndarray(shape= (indi, var), dtype=('U', len_v))
    tmp_pop2 = [[makeGen(len_v) for x in tmp_pop[0]] for y in tmp_pop]
    return(np.array(tmp_pop2))

def annularCross(I_double, n, len_v, n_vars, Pc):
    # This function apply Deterministic Annular Crossover to the population
    I_tmp = np.copy(I_double)
    chromosome = len_v * n_vars
    cross_prob = np.random.random(1)
    for count in range(int(n/4)):
        if(cross_prob <= Pc):
            cut1 = np.random.randint(0, chromosome-1)
            len_cut = np.random.randint(0, int(chromosome/2))
            cut2= (cut1 + len_cut) % chromosome
            tmp_str1 = functools.reduce(lambda x, y: x + y, I_double[count])
            tmp_str2 = functools.reduce(lambda x, y: x + y, I_double[int(n/2) - count -1])
            if(cut2 < cut1):
                cut2, cut1 = cut1, cut2
            tmp_str1, tmp_str2 = tmp_str1[0:cut1]+tmp_str2[cut1:cut2]+tmp_str1[cut2:], tmp_str2[0:cut1]+tmp_str1[cut1:cut2]+tmp_str2[cut2:]
            init = 0
            end = len_v
            for i in range(n_vars):
                I_tmp[count][i] = tmp_str1[init:end]
                I_tmp[int(n/2) - count -1][i] = tmp_str2[init:end]
                init += len_v
                end += len_v
    return I_tmp

def mutation(I_double, n, len_v, n_var, B2M):
    I_tmp = np.copy(I_double)
    for count in range(B2M):
        p1 = int(np.random.random(1) * int(n/2))
        p2 = int(np.random.random(1) * (len_v*n_vars))
        aff_var = int(p2/len_v)
        aff_alle = int(p2%len_v)
        tmp = list(I_tmp[p1][aff_var])
        if(I_tmp[p1][aff_var][aff_alle:aff_alle+1] == '1'):
            tmp[aff_alle] = '0'
        else:
            tmp[aff_alle] = '1'
        I_tmp[p1][aff_var] = ''.join(tmp)
    return (I_tmp)

### Variables for flexibility of the algorithm
# number of variables for one solution
n_vars = 42
# lenght of bits for each variable
len_v = 24
### Variables what EGA needs
# Number of generations
G = 20
# Number of individuals
n = 3
# Length of chromosome
L = n_vars * len_v
# Population
I = np.ndarray(shape=(n, n_vars), dtype=('U', len_v))
# Crossing probability
Pc = 0.9
# Mutation probability
Pm = 0.05
# list of fitness
fitness = np.ndarray(shape= (2, n), dtype=float)
# Expected number of mutations for each generation
B2M = int(n * L * Pm)

### Auxiliary variables
# Double of the population
I_double = np.ndarray(shape=(2*n, n_vars), dtype= ('U', len_v))
# double of the list of fitness
fitness_double = np.ndarray(shape=(2, 2*n), dtype=float)


# Initial population
I = genInitPop(n, n_vars, len_v)

for gen in range(G):
    # Double of length of the population
    I_double = np.concatenate((I, I), axis=0)

    # Apply Annular Crossover
    I_double = annularCross(I_double, len(I_double), len_v, n_vars, Pc)

    # Apply Mutation
    I_double = mutation(I_double, len(I_double), len_v, n_vars, B2M)

    # Apply fitness
    count = 0
    for i in range(fitness_double.shape[1]):
        fitness_double[0][i] = count
        fitness_double[1][i] = fnn.fitness(I_double[i])
        count += 1
    # Order by fitness
    fitness_double = fitness_double[:, fitness_double[1].argsort()]
    print(fitness_double)
    fitness_double = list(fitness_double)
    fitness_double = fitness_double[::-1]
    fitness_double = np.array(fitness_double)
    print(fitness_double)
    # Apply Elitism
    ind_eli = fitness_double[0][0:n]
    ind_eli = np.array(ind_eli, dtype='int32')
    count = 0
    for i in ind_eli:
        I[count] = I_double[i]
        count += 1
    print("Generation {}, MSE= {}".format(gen+1, fitness_double[1][0]))
print("Aproaches: ")
print('Pesos: ')
for i in range(len(I[0])):
    print(fnn.bin2float(I[i], 1))
print("Error promedio= ", fnn.fitness(I[0]))

