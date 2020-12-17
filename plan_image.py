import numpy as np
import argparse
import random

from pathlib import Path
from datetime import datetime

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

from classloader import load_image_function

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

#funcao de fitness
def f_fitness(individual):
    return np.sum(individual),

#funcao de crossover
def cxTwoPointCopy(ind1, ind2):
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
        
    return ind1, ind2
    
# Registar funcao de fitness, crossover, mutacao e selecao
toolbox.register("evaluate", f_fitness)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)

toolbox.register("clip", np.clip, a_min=0.02, a_max=0.98)
toolbox.register("select", tools.selTournament, tournsize=5)

# Funcao para preencher os individuos
toolbox.register("attr_float", np.random.uniform, 0.02, 0.98)

# Subdividir os arrays numpy em arrays de arrays para conseguir fazer o rendering
def chunks(array, num_cols):
    return [ array[i:i+num_cols] for i in range(0, len(array), num_cols) ]


def check_bounds(offspring, lower_bound, upper_bound):
    for child in offspring:
        for i in range(len(child)):
            if child[i] > upper_bound:
                child[i] = upper_bound
            elif child[i] < lower_bound:
                child[i] = lower_bound
    
    return offspring


def optimize(outdir, render_image, render_size, max_attempts, max_iterations, num_pop, num_lines, num_cols):
    # Criacao da populacao inicial
    pop = toolbox.population(n=num_pop)
    best = None
    best_fitness = None

    # Calcular as varias fitnesses e guardar a melhor
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    index = 0
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

        if (best == None or fit > best_fitness):
            best = index
            best_fitness = fit
        
        index += 1

    # Guardar como imagem o primeiro best
    im_array = chunks(pop[best], num_cols)
    im = render_image(im_array, size=render_size)
    im.save('{}/start.png'.format(outdir))
    iters_since_best = 0

    # Ciclo principal
    for it in range(max_iterations):
        im = render_image(im_array, size=render_size)
        im.save('{}/epoch_{}.png'.format(outdir, it))

        # Criacao de uma nova populacao
        offspring = toolbox.select(pop, len(pop))
        offspring = [ toolbox.clone(ind) for ind in offspring ]

        #MUDAR ISTO
        CROSSOVER_PROB = 0.9
        MUTATION_PROB = 0.1
        offspring = algorithms.varAnd(offspring, toolbox,CROSSOVER_PROB, MUTATION_PROB)
        # # Aplicar crossover
        # for child1, child2 in zip(offspring[::2], offspring[1::2]):
        #     if random.random() < 0.5:
        #         toolbox.mate(child1, child2)
        
        # # Aplicar mutacao
        # for mutant in offspring:
        #     if random.random() < 0.2:
        #         toolbox.mutate(mutant)

        
        #pop = check_bounds(offspring, 0.02, 0.98)
        offspring = toolbox.clip(offspring)
        

        # Calcular fitness e guardar o melhor da nova populacao
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        index = 0
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

            if (fit > best_fitness):
                best = index
                best_fitness = fit
                iters_since_best = 0
            else:
                iters_since_best += 1
            
            index += 1
        
        im_array = chunks(pop[best], num_cols)
    
    print(best_fitness)



def main():
    # Parsing dos argumentos de input
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-lines', default=17, type=int)
    parser.add_argument('--num-cols', default=8, type=int)
    parser.add_argument('--num-pop', default=100, type=int)
    parser.add_argument('--max-iterations', default=1000, type=int)
    parser.add_argument('--max-attempts', default=30, type=int)
    parser.add_argument('--random-seed', default=None)
    parser.add_argument('--renderer', default='lines1')
    parser.add_argument('--render-size', default=512, type=int)
    parser.add_argument('--outdir', default='outputs')

    args = parser.parse_args()

    # Definicao dos individuos e populacao 
    # Um array de tamanho num_lines*num_cols 
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=args.num_lines*args.num_cols)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Funcao de rendering
    render_image = load_image_function(args.renderer + '.render')

    # Criar a diretoria para guardar as varias imagens e logs
    # Se seed == None: O path fica com o timestamp
    outdir = args.outdir + '/L{}_C{}_{}'.format(
        args.num_lines, 
        args.num_cols, 
        args.random_seed if args.random_seed else datetime.now().strftime('%Y-%m-%d_%H-%M') 
    )
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Override da seed. Se for None, default e o timestamp
    random.seed(args.random_seed)

    optimize(outdir, render_image, args.render_size, args.max_attempts, args.max_iterations, args.num_pop, args.num_lines, args.num_cols)

if __name__ == "__main__":
    main()