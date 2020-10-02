import covasim as cv
import pandas as pd
import sciris as sc

import population


def make_people(seed, pop_size):
    mixing_H = pd.read_csv('data/mixing_H.csv', index_col='Age group')
    reference_ages = pd.read_csv('data/reference_ages.csv', index_col='age', squeeze=True)
    households = pd.read_csv('data/households.csv', index_col='size', squeeze=True)
    layers = pd.read_csv('layers.csv', index_col='layer')

    print('TODO - asymmetric mixing matrix')
    mixing_H = (mixing_H + mixing_H.T) / 2
    cv.set_seed(seed)

    # First, generate the household layer (produces a cv.People object with one layer only)
    people = population.generate_people(int(pop_size), mixing_H, reference_ages, households)

    # Then, add the remaining layers to the cv.People object
    # In particular, the school and work functions could easily be rewritten for a different setting/analysis
    population.add_school_contacts(people, mean_contacts=layers.loc['S', 'contacts'])
    population.add_work_contacts(people, mean_contacts=layers.loc['W', 'contacts'])
    population.add_other_contacts(people, layers)

    return people


if __name__ == '__main__':
    with sc.Timer(label='Make people'):
        people = make_people(seed=1, pop_size=100e3)
        # sc.saveobj('nswppl.pop', people)

old = sc.loadobj('nswppl.pop')