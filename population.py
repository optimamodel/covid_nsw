# FUNCTIONS TO CREATE POPULATIONS

from collections import defaultdict

import covasim as cv
import covasim.defaults as cvd
import covasim.utils as cvu
import covasim_victoria as cvv
import numba as nb
import numpy as np
import pandas as pd


## MAIN API

def generate_people(n_people: int, mixing: pd.DataFrame, reference_ages: pd.Series, households: pd.Series) -> cv.People:
    """
    Construct a cv.People object

    This will construct a cv.People based on demographics, and a household layer based on the mixing matrix.
    These are done together because it's simpler to populate households based on the mixing matrix until a
    until a sufficient number of agents have been created, rather than creating the agents first and then
    trying to assign them to households to match the mixing matrix.

    Args:
        n_people: Number of simulation agents to create
        mixing: The mixing matrix dataframe (obtained by loading the mixing matrix CSV file)
        reference: Distribution of reference ages. This should be a pd.Dataframe/pd.Series where the index is the age, and the value is the (relative or absolute) number of people of that age
        households: Distribution of households of different sizes. Should be a pd.Dataframe/pd.Series where the index is the number of people in the household, and the value is the (relative or absolute) number of households of that size.

    Returns: - A `cv.People` instance with a household contact layer

    """

    # First work out how many households of each size we need
    total_people = sum(households.index * households.values)  # total_people = household_size * n_households
    household_percent = households / total_people
    n_households = (n_people * household_percent).round().astype(int)
    n_households[1] += n_people - sum(n_households * n_households.index)  # adjust single-person households to fill the gap

    # Then, select a reference person age for the first person in each household
    household_heads = np.random.choice(reference_ages.index, size=sum(n_households), p=reference_ages.values / sum(reference_ages))

    h_clusters, ages = _make_households(n_households, n_people, household_heads, mixing)

    contacts = cv.Contacts()
    contacts['H'] = cvv.clusters_to_layer(h_clusters)
    people = cv.People(pars={'pop_size': n_people}, age=ages)
    people.contacts = contacts

    return people


def add_school_contacts(people: cv.People, mean_contacts: float):
    """Create school contacts, with children of each age clustered in groups"""

    classrooms = []

    for age in range(5, 18):
        children_thisage = cvu.true(people.age == age)
        classrooms.extend(cvv.create_clusters(children_thisage, mean_contacts))

        teachers = np.random.choice(cvu.true(people.age > 21), len(classrooms), replace=False)
        for i in range(len(classrooms)):
            classrooms[i].append(teachers[i])

    people.contacts['S'] = cvv.clusters_to_layer(classrooms)


def add_work_contacts(people: cv.People, mean_contacts: float):
    work_inds = cvu.true((people.age > 18) & (people.age <= 65))
    work_cl = cvv.create_clusters(work_inds, mean_contacts)
    people.contacts['W'] = cvv.clusters_to_layer(work_cl)


def add_other_contacts(people: cv.People, layers: pd.DataFrame, legacy=True):
    """
    Add layers according to a layer file

    Args:
        people: A cv.People instance to add new layers to
        layer_members: Dict containing {layer_name:[indexes]} specifying who is able to have interactions within each layer
        layerfile: Dataframe from `layers.csv` where the index is the layer name

    """

    for layer_name, layer in layers.iterrows():

        if layer['cluster_type'] in {'home', 'school', 'work'}:
            # Ignore these cluster types, as they should be instantiated with
            # - home: make_people()
            # - school: add_school_contacts()
            # - work: add_work_contacts()
            continue

        age_min = 0 if pd.isna(layer['age_lb']) else layer['age_lb']
        age_max = np.inf if pd.isna(layer['age_ub']) else layer['age_ub']
        age_eligible = cvu.true((people.age >= age_min) & (people.age <= age_max))
        n_people = int(layer['proportion'] * len(age_eligible))
        inds = np.random.choice(age_eligible, n_people, replace=False)

        if layer['cluster_type'] == 'cluster':
            # Create a clustered layer based on the mean cluster size
            assert pd.isna(layer['dynamic']), 'Dynamic clusters not supported yet'
            clusters = cvv.create_clusters(inds, layer['contacts'])
            people.contacts[layer_name] = cvv.clusters_to_layer(clusters)
        elif layer['cluster_type'] == 'complete':
            # For a 'complete' layer, treat the layer members as a single cluster
            assert pd.isna(layer['dynamic']), 'Dynamic complete clusters not supported yet'
            people.contacts[layer_name] = cvv.clusters_to_layer([inds])
        elif layer['cluster_type'] == 'random':
            people.contacts[layer_name] = cvv.RandomLayer(inds, layer['contacts'], layer['dispersion'], dynamic=(not pd.isna(layer['dynamic'])))
        else:
            raise Exception(f'Unknown clustering type {layer["cluster_type"]}')


## EXTERNALLY-FACING HELPERS

class RandomLayer(cv.Layer):
    """
    Layer that can resample contacts on-demand
    """

    def __init__(self, inds, mean_contacts, dispersion=None, dynamic=False):
        """

        Args:
            inds:
            mean_contacts:
            dispersion: Level
            dynamic: If True, the layer will change each timestep
        """
        super().__init__()
        self.inds = inds
        self.mean_contacts = mean_contacts
        self.dispersion = dispersion
        self.dynamic = dynamic
        self.update(force=True)

    @staticmethod
    @nb.njit
    def _get_contacts(inds, number_of_contacts):
        """
        Efficiently generate contacts

        Note that because of the shuffling operation, each person is assigned 2N contacts
        (i.e. if a person has 5 contacts, they appear 5 times in the 'source' array and 5
        times in the 'target' array). This is why `clusters_to_layer` must add bidirectional
        contacts as well, so that all contacts are consistently specified bidirectionally.

        Args:
            inds: List/array of person indices
            number_of_contacts: List/array the same length as `inds`

        Returns: Two arrays, for source and target


        """
        total_number_of_half_edges = np.sum(number_of_contacts)

        count = 0
        source = np.zeros((total_number_of_half_edges,), dtype=cvd.default_int)
        for i, person_id in enumerate(inds):
            n_contacts = number_of_contacts[i]
            source[count:count + n_contacts] = person_id
            count += n_contacts
        target = np.random.permutation(source)

        return source, target

    def update(self, force: bool = False) -> None:
        """
        Regenerate contacts

        Args:
            force: If True, ignore the `self.dynamic` flag. This is required for initialization.

        """

        if not self.dynamic and not force:
            return

        n_people = len(self.inds)

        # sample the number of edges from a given distribution
        if pd.isna(self.dispersion):
            number_of_contacts = cvu.n_poisson(rate=self.mean_contacts, n=n_people)
        else:
            number_of_contacts = cvu.n_neg_binomial(rate=self.mean_contacts, dispersion=self.dispersion, n=n_people)

        self['p1'], self['p2'] = self._get_contacts(self.inds, number_of_contacts)
        self['beta'] = np.ones(len(self['p1']), dtype=cvd.default_float)
        self.validate()


class UpdateNetworks(cv.Intervention):
    """
    Intervention to update dynamic layers

    This should be added if any RandomLayer instances are present.
    It triggers re-sampling the contacts each timestep
    """

    def apply(self, sim):
        for layer in sim.people.contacts.values():
            if isinstance(layer, RandomLayer):
                layer.update()


def create_clusters(people_to_cluster: list, mean_cluster_size: float) -> list:
    """
    Assign people to clusters

    Returns a list of clusters (suitable for `clusters_to_layer`)

    Args:
        people_to_cluster: Indexes of people to cluster e.g. [1,5,10,12,13]
        mean_cluster_size: Mean cluster size (poisson distribution)

    Returns: List of lists of clusters e.g. [[1,5],[10,12,13]]
    """

    # people_to_cluster = np.random.permutation(people_to_cluster) # Optionally shuffle people to cluster - in theory not necessary?
    clusters = []
    n_people = len(people_to_cluster)
    n_remaining = n_people

    while n_remaining > 0:
        this_cluster = cvu.poisson(mean_cluster_size)  # Sample the cluster size
        if this_cluster > n_remaining:
            this_cluster = n_remaining
        clusters.append(people_to_cluster[(n_people - n_remaining) + np.arange(this_cluster)].tolist())
        n_remaining -= this_cluster

    return clusters


def clusters_to_layer(clusters: list):
    """
    Convert a list of clusters to a Covasim layer

    Assumes fully connected clusters. Note that in Covasim-Victoria, *all* contacts added
    to Covasim should be bidirectional. It's possible that a person could belong to more than one cluster

    Args:
        clusters: List of lists containing cluster members e.g. ``[[1,2,3],[4],[5,6]]``

    Returns: - A layer with fully connected contacts e.g. ``1-2,2-1,1-3,3-1 2-3,3-2,5-6,6-5`` for the example above
             - An array of person IDs (indexes) corresponding to the people in the layer
    """

    p1 = []
    p2 = []
    layer_members = set()

    for cluster in clusters:
        for i in cluster:
            layer_members.add(i)
            for j in cluster:
                if j != i:
                    p1.append(i)
                    p2.append(j)

    layer = cv.Layer()
    layer['p1'] = np.array(p1, dtype=cvd.default_int)
    layer['p2'] = np.array(p2, dtype=cvd.default_int)
    layer['beta'] = np.ones(len(p1), dtype=cvd.default_float)
    layer.validate()
    # layer.members = np.array(sorted(layer_members), dtype=cvd.default_int)

    return layer


## INTERNAL HELPERS FOR HOUSEHOLD GENERATION

## Fast choice implementation
# From https://gist.github.com/jph00/30cfed589a8008325eae8f36e2c5b087
# by Jeremy Howard https://twitter.com/jeremyphoward/status/955136770806444032
@nb.njit
def sample(n, q, J, r1, r2):
    res = np.zeros(n, dtype=np.int32)
    lj = len(J)
    for i in range(n):
        kk = int(np.floor(r1[i] * lj))
        if r2[i] < q[kk]:
            res[i] = kk
        else:
            res[i] = J[kk]
    return res


class AliasSample():
    def __init__(self, probs):
        self.K = K = len(probs)
        self.q = q = np.zeros(K)
        self.J = J = np.zeros(K, dtype=np.int)

        smaller, larger = [], []
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small, large = smaller.pop(), larger.pop()
            J[small] = large
            q[large] = q[large] - (1.0 - q[small])
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

    def draw_one(self):
        K, q, J = self.K, self.q, self.J
        kk = int(np.floor(np.random.rand() * len(J)))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]

    def draw_n(self, n):
        r1, r2 = np.random.rand(n), np.random.rand(n)
        return sample(n, self.q, self.J, r1, r2)


def _sample_household_cluster(sampler, bin_lower, bin_upper, reference_age, n):
    """
    Return list of ages in a household/location based on mixing matrix and reference person age
    """

    ages = [reference_age]  # The reference person is in the household/location

    if n > 1:
        idx = np.digitize(reference_age, bin_lower) - 1  # First, find the index of the bin that the reference person belongs to
        sampled_bins = sampler[idx].draw_n(n - 1)

        for bin in sampled_bins:
            ages.append(int(round(np.random.uniform(bin_lower[bin] - 0.5, bin_upper[bin] + 0.5))))

    return np.array(ages)


def _make_households(n_households, pop_size, household_heads, mixing_matrix):
    """

    The mixing matrix is a direct read of the CSV file, with index corresponding to 'Age group' i.e.

    >>> mixing_matrix = pd.read_csv('mixing_H.csv',index_col='Age group')
    >>> mixing_matrix
                   0 to 4    5 to 9
        Age group
        0 to 4     0.659868  0.503965
        5 to 9     0.314777  0.895460


    :param n_households:
    :param pop_size:
    :param household_heads:
    :return:
        h_clusters: a list of lists in which each sublist contains
                    the IDs of the people who live in a specific household
        ages: flattened array of ages, corresponding to the UID positions
    """
    mixing_matrix = mixing_matrix.div(mixing_matrix.sum(axis=1), axis=0)
    samplers = [AliasSample(mixing_matrix.iloc[i, :].values) for i in range(mixing_matrix.shape[0])]  # Precompute samplers for each reference age bin

    age_lb = [int(x.split(' to ')[0]) for x in mixing_matrix.index]
    age_ub = [int(x.split(' to ')[1]) for x in mixing_matrix.index]

    h_clusters = []
    uids = np.arange(0, pop_size)
    ages = np.zeros(pop_size, dtype=int)
    h_added = 0
    p_added = 0

    for h_size, h_num in n_households.iteritems():
        for household in range(h_num):
            head = household_heads[h_added]
            # get ages of people in household
            household_ages = _sample_household_cluster(samplers,
                                                       age_lb,
                                                       age_ub,
                                                       head,
                                                       h_size)
            # add ages to ages array
            ub = p_added + h_size
            ages[p_added:ub] = household_ages
            # get associated UID that defines a household cluster
            h_ids = uids[p_added:ub]
            h_clusters.append(h_ids)
            # increment sliding windows
            h_added += 1
            p_added += h_size
    return h_clusters, ages
