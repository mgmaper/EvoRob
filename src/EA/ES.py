import os
from typing import Dict

import numpy as np

from src.utils.Filesys import search_file_list

ES_opts = {
    "min": -4,
    "max": 4,
    "num_parents": 16,
    "num_generations": 100,
    "mutation_sigma": 0.3,
    "sigma_limit": 0.1,
}


class ES:

    def __init__(self, n_pop, n_params, opts: Dict=ES_opts, output_dir: str="./results/ES"):
        """
        Evolutionary Strategy [INCOMPLETE]

        :param n_pop: population size
        :param n_params: number of parameters
        :param opts: algorithm options
        :param output_dir: output directory Default = "./results/ES"
        """
        self.n_params = n_params
        self.n_pop = n_pop
        self.n_gen = opts["num_generations"]
        self.n_parents = opts["num_parents"]
        self.min = opts["min"]
        self.max = opts["max"]

        self.current_gen = 0
        self.current_mean = self.initialise_x0()  #TODO
        self.current_sigma = opts["mutation_sigma"]
        self.sigma_limit = opts["sigma_limit"]

        #% bookkeeping
        self.directory_name = output_dir
        self.full_x = []
        self.full_fitness = []
        self.x_best_so_far = None
        self.f_best_so_far = -np.inf
        self.x = None
        self.f = None

    def ask(self):
        if self.current_gen==0:
            new_population = self.initialise_x0()
        else:
            new_population = self.generate_mutated_offspring(self.n_pop)
        new_population = np.clip(new_population, self.min, self.max)
        return new_population

    def tell(self, solutions, function_values, save_checkpoint=True):
        parents_population, parents_fitness = self.sort_and_select_parents(
            solutions, function_values, self.n_parents
        )
        self.current_mean = self.update_population_mean(parents_population, parents_fitness)
        self.current_sigma = self.update_sigma()


        #% Some bookkeeping
        self.full_fitness.append(function_values)
        self.full_x.append(solutions)
        self.x = parents_population
        self.f = parents_fitness

        if np.max(function_values) > self.f_best_so_far:
            best_index = np.argmax(function_values)
            self.f_best_so_far = function_values[best_index]
            self.x_best_so_far = solutions[best_index]

        if self.current_gen % 5 == 0:
            print(f"Generation {self.current_gen}:\t{self.f_best_so_far}\n"
                  f"Mean fitness:\t{self.f.mean()} +- {self.f.std()}\n"
                  f"Sigma: {self.current_sigma} \n"
                  )

        if save_checkpoint:
            self.save_checkpoint()
        self.current_gen += 1

    def initialise_x0(self,):
        #TODO
        mean_vector = np.random.uniform(self.min, self.max, (self.n_pop, self.n_params))

        return mean_vector

    def generate_mutated_offspring(self, population_size):
        # TODO
        population = np.tile(self.current_mean, (population_size, 1))

        # Compute multivariate Gaussian noise
        mutation = np.random.normal(0, 1, size = (population_size, len(self.current_mean)))

        # Compute offspring
        mutated_population = population + self.current_sigma * mutation

        return mutated_population

    def sort_and_select_parents(self, population, fitness, num_parents):
        # TODO
        sorted_indices = np.argsort(fitness)[::-1]
        sorted_indices = sorted_indices[0:num_parents]

        parent_population = population[sorted_indices]
        parent_fitness = fitness[sorted_indices]

        return parent_population, parent_fitness

    def update_population_mean(self, parent_population, parent_fitness):
        # TODO
        # Normalise parent fitness scores
        #normed_parents_fitness = parent_fitness / np.sum(parent_fitness) #Not the same as in the solution
        normed_fitness = (parent_fitness - parent_fitness[-1])/(parent_fitness[0]-parent_fitness[-1])
        normed_parents_fitness = normed_fitness / np.sum(normed_fitness)
        
        # Compute population weighted to the normed fitness scores
        weight = np.outer(normed_parents_fitness, np.ones((1, parent_population.shape[1])))
        weighted_parents_population = np.multiply(
            parent_population, weight
        )  # hadamard product

        # Calculate the sum of weighted parents population
        updated_mean_vector = np.sum(weighted_parents_population, axis=0)


        return updated_mean_vector

    def update_sigma(self):
        #TODO
        sigma_limit = self.sigma_limit
        sigma = self.current_sigma
        param_size = self.n_params
        tau = 1 / (np.sqrt(param_size))
        sigma = sigma * np.exp(tau * np.random.normal())
        if sigma < sigma_limit :
            sigma = sigma_limit
        return sigma

    def save_checkpoint(self):
        curr_gen_path = os.path.join(self.directory_name, str(self.current_gen))
        os.makedirs(curr_gen_path, exist_ok=True)
        np.save(os.path.join(self.directory_name, 'full_f'), np.array(self.full_fitness))
        np.save(os.path.join(self.directory_name, 'full_x'), np.array(self.full_x))
        np.save(os.path.join(curr_gen_path, 'f_best'), np.array(self.f_best_so_far))
        np.save(os.path.join(curr_gen_path, 'x_best'), np.array(self.x_best_so_far))
        np.save(os.path.join(curr_gen_path, 'x'), np.array(self.x))
        np.save(os.path.join(curr_gen_path, 'f'), np.array(self.f))

    def load_checkpoint(self):
        dir_path = search_file_list(self.directory_name, 'f_best.npy')
        assert len(dir_path) > 0;
        "No files are here, check the directory_name!!"

        self.current_gen = int(dir_path[-1].split('/')[-2])
        curr_gen_path = os.path.join(self.directory_name, str(self.current_gen))

        self.full_fitness = np.load(os.path.join(self.directory_name, 'full_f.npy'))
        self.full_x = np.load(os.path.join(self.directory_name, 'full_x.npy'))
        self.f_best_so_far = np.load(os.path.join(curr_gen_path, 'f_best.npy'))
        self.x_best_so_far = np.load(os.path.join(curr_gen_path, 'x_best.npy'))
        self.x = np.load(os.path.join(curr_gen_path, 'x.npy'))
        self.f = np.load(os.path.join(curr_gen_path, 'f.npy'))

