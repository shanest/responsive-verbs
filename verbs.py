"""
Copyright (C) 2018 Shane Steinert-Threlkeld

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
"""
# TODO: document
import abc
from functools import reduce
import random
import numpy as np


""" The following code for generating random partitions of a given number of
cells of a set has been adapted from the following StackOverflow answer by
Peter de Rivaz:
    https://stackoverflow.com/a/45885244/9370349
"""
# TODO: better variable names below?
fact = [1]


def num_choose_k(n, k):
    """Get number of ways of choosing k elements from n.

    Args:
        n: size of set to choose from
        k: number of elements

    Returns:
        size of (n choose k)
    """
    while len(fact) <= n:
        fact.append(fact[-1] * len(fact))
    return fact[n] / (fact[k] * fact[n - k])


cache = {}


def count_partitions(n, k):
    """Count number of ways of partitioning n items into k non-empty subsets.
    This is the second Stirling number S(n, k).
    """
    if k == 1:
        return 1

    key = n, k
    if key in cache:
        return cache[key]

    # The first element goes into the next partition
    # We can have up to y additional elements from the n-1 remaining
    # There will be n-1-y left over to partition into k-1 non-empty subsets
    # so n-1-y>=k-1
    # y<=n-k
    t = 0
    for y in range(0, n - k + 1):
        t += count_partitions(n - 1 - y, k - 1) * num_choose_k(n - 1, y)
    cache[key] = t
    return t


def ith_subset(A, k, i):
    """Return ith k-subset of A"""
    n = len(A)
    if n == k:
        return tuple(A)
    if k == 0:
        return tuple([])
    # Choose first element x
    for x in range(n):
        # Find how many cases are possible with the first element being x
        # There will be n-x-1 left over, from which we choose k-1
        extra = num_choose_k(n - x - 1, k - 1)
        if i < extra:
            break
        i -= extra
    return (A[x],) + ith_subset(A[x + 1 :], k - 1, i)


def gen_part(A, k, i):
    """Return i^th k-partition of elements in A (zero-indexed) as
    list of lists"""
    if k == 1:
        return (tuple(A),)

    n = len(A)
    # First find appropriate value for y - the extra amount in this subset
    for y in range(0, n - k + 1):
        extra = count_partitions(n - 1 - y, k - 1) * num_choose_k(n - 1, y)
        if i < extra:
            break
        i -= extra

    # We count through the subsets,
    # and for each subset we count through the partitions
    # Split i into a count for subsets and a count for the remaining partitions
    count_partition, count_subset = divmod(i, num_choose_k(n - 1, y))
    # Now find the i^th appropriate subset
    subset = (A[0],) + ith_subset(A[1:], y, count_subset)
    S = set(subset)
    return (subset,) + gen_part([a for a in A if a not in S], k - 1, count_partition)


""" End adaptation of random partition code from de Rivaz. """


def embedding(partition, num_worlds):
    """This function implements the E function from Theiler, Aloni, and
    Roelofsen.  In particular, it returns a square matrix of size num_worlds,
    with each row of the matrix corresponding to the complete answer to the
    question represented by partition in that world.

    See module doc-string for notational conventions used throughout.

    Args:
        partition: a list of lists

    Returns:
        matrix of shape (num_worlds, num_worlds)
    """
    result = np.zeros((num_worlds, num_worlds))
    for cell in partition:
        for world in cell:
            result[world, cell] = 1
    return result


def embedding2(partition, num_worlds, max_cells):
    result = np.zeros((max_cells, num_worlds))
    for idx in range(len(partition)):
        cell = partition[idx]
        result[idx, cell] = 1
    return result


def generate_partition(num_worlds, is_declarative, max_cells=5):
    """Generates a partition of a set of size num_worlds.  The partition will
    be a list of lists, with each element of range(num_worlds) appearing in
    exactly one such list.

    Args:
        num_worlds: plays two roles. (i) length of partition, (ii) maximum
                    number of possible cells
        is_declarative: boolean
        max_cells: maximum number of cells to include in partition

    Returns:
        a list of lists, corresponding to a partition of range(num_worlds)
        If is_declarative: the partition will only have one cell, corresponding
        to info(P)
    """
    num_cells = 2 if is_declarative else np.random.randint(2, max_cells + 1)
    part_index = np.random.randint(count_partitions(num_worlds, num_cells))
    partition = gen_part(range(num_worlds), num_cells, part_index)
    if is_declarative:
        # declaratives only have one cell, i.e. one alternative
        which_cell = np.random.randint(2)
        partition = (partition[which_cell],)
    # TODO: if needed, return num_cells and part_index, to rule out duplicates
    # in data generator
    return partition


def in_partition(part, elt):
    """Whether an element is in a partition.

    Args:
        part: list of lists
        elt: an element


    Returns:
        True if elt is in one of the lists in part
    """
    for cell in part:
        if elt in cell:
            return True
    return False


def complement(part, num_worlds):
    """Complement of a list of lists, with respect to range(num_worlds).  If
    the list of lists is a genuine partition, this will return a tuple
    containing the empty tuple.  Otherwise, a length-1 tuple with the
    complement worlds in it.

    This can be seen as inquisitive negation.

    Args:
        part: partition, list of lists
        num_worlds: range(num_worlds) is the set of worlds for complementing

    Returns:
        a length 1 tuple, with a tuple of all worlds not in part
    """
    worlds = range(num_worlds)
    in_part = set([item for cell in part for item in cell])
    return (tuple([world for world in worlds if world not in in_part]),)


def fill_in_partition(part, num_worlds):
    """Turn a declarative meaning P into the polar question ?P.

    Args:
        part: a partition
        num_worlds: number of worlds

    Returns:
        part if complement is trivial, (part, complement[part]) otherwise
    """
    comp = complement(part, num_worlds)
    return part + comp if len(comp[0]) > 0 else part


def cell_of_elt(part, elt, num_worlds=None):
    """Get the cell of an element.  If part is a declarative meaning and elt is
    not in part, then this returns the complement of part.

    Args:
        part: partition
        elt: element
        num_worlds: num_worlds

    Returns:
        the cell of the partition containing elt, or complement(part,
        num_worlds)
    """
    for cell in part:
        if elt in cell:
            return cell
    return complement(part, num_worlds)[0]


def index_of_elt(part, elt):
    """Get the index of the cell containing an element.

    Args:
        part: partition
        elt: element

    Returns:
        i such that elt is in part[i]

    Raises:
        ValueError, if elt is not in part
    """
    for idx in range(len(part)):
        if elt in part[idx]:
            return idx
    raise ValueError("{} not in {}".format(elt, part))


def intersecting_cells(partition, dox_w):
    """ Get all cells of partition that intersect with dox_w. """
    cells = set()
    num_worlds = len(dox_w)
    partition = fill_in_partition(partition, num_worlds)
    for world in range(num_worlds):
        if dox_w[world]:
            cells.add(partition[index_of_elt(partition, world)])
    return cells


def num_cells_intersect(partition, dox_w):
    """How many cells of a partition does a set of worlds non-trivially
    intersect.

    Args:
        partition: partition
        dox_w: a binary vector

    Returns:
        the number of items in partition containing a world such that
        dox_w[world] is 1
    """
    return len(intersecting_cells(partition, dox_w))


def list_subset(ls1, ls2):
    """Whether one list is a subset of another. """
    return set(ls1) <= set(ls2)


def flatten_partition(partition):
    return reduce(lambda a, b: a + b, partition)


class Verb(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def generate_true(num_worlds, max_cells):
        """Generate true examples of your Verb.

        Args:
            num_worlds: how many worlds

        Returns:
            partition: tuple of tuples, a partition
            world: int, the actual world
            dox_w: 1-D array of 1s and 0s, the agent's doxastic state
        """
        pass

    @abc.abstractmethod
    def verify_true(partition, world, dox_w, is_declarative):
        """Verify truth of a tuple for the verb.

        Args:
            partition: tuple of tuples, a partition
            world: int, the actual world
            dox_w: 1-D array of 1s and 0s, the agent's doxastic state
            is_declarative: boolean, whether declarative or not

        Returns:
            boolean, whether verb holds or not
        """
        pass

    @classmethod
    def generate(cls, num_worlds, truth_value, max_cells=5):
        """Generate examples.  Calls generate_true or generate_false based on
        the argument truth_value.
        """
        return (
            cls.generate_true(num_worlds, max_cells)
            if truth_value
            else cls.generate_false(num_worlds, max_cells)
        )

    @classmethod
    def generate_false(cls, num_worlds, max_cells):
        """Generate false examples of your Verb.

        Args:
            num_worlds: how many worlds
            max_cells: larges number of cells to generate

        Returns:
            partition: tuple of tuples, a partition
            world: int, the actual world
            dox_w: 1-D array of 1s and 0s, the agent's doxastic state
        """
        partition, world, dox_w, is_declarative = Verb.initialize(
            num_worlds, max_cells, dox_random=True
        )

        while cls.verify_true(partition, world, dox_w, is_declarative):
            partition, world, dox_w, is_declarative = Verb.initialize(
                num_worlds, max_cells, dox_random=True
            )

        return partition, world, dox_w

    @staticmethod
    def initialize(num_worlds, max_cells, dox_random=False):
        """Perform initial prep for generate functions.

        Args:
            num_worlds: how many worlds
            max_cells: largest number of cells to generate
            dox_random: whether dox_w should be random or all zeros

        Returns:
            partition: generate_partition(num_worlds, is_declarative)
            world: a random integer up to num_worlds
            dox_w: a 1-D zero array of length num_worlds
            is_declarative: boolean, whether it's a declarative example or not
        """
        is_declarative = np.random.random() < 0.5
        partition = generate_partition(num_worlds, is_declarative, max_cells)
        world = np.random.randint(num_worlds)
        dox_w = np.zeros([num_worlds], dtype=np.int_)
        if dox_random:
            # make sure dox_w is non-empty if doing random generation
            while sum(dox_w) == 0:
                dox_w = np.random.choice([0, 1], [num_worlds])
        return partition, world, dox_w, is_declarative


class Know(Verb):
    """Verb meaning: \P \w: dox_w in P and w in dox_w
    """

    @staticmethod
    def generate_true(num_worlds, max_cells):

        partition, world, dox_w, is_declarative = Verb.initialize(num_worlds, max_cells)

        if is_declarative and not in_partition(partition, world):
            # proposition has to be true at w!
            partition = ((partition[0] + (world,)),)

        # get the true answer at w
        world_cell = cell_of_elt(partition, world, num_worlds)
        # randomly include worlds from Q_w
        how_many = 1 + np.random.randint(len(world_cell))
        dox_w[np.random.choice(world_cell, [how_many], replace=False)] = 1

        dox_w[world] = 1

        return partition, world, dox_w

    @staticmethod
    def verify_true(partition, world, dox_w, is_declarative):

        veridical = in_partition(partition, world) and dox_w[world] == 1

        world_cell = cell_of_elt(partition, world, len(dox_w))
        dox_cell = np.nonzero(dox_w)[0]
        dox_sub_w = list_subset(dox_cell, world_cell)

        return veridical and dox_sub_w


class BeCertain(Verb):
    """Verb meaning: \P \w: dox_w in P
    """

    @staticmethod
    def generate_true(num_worlds, max_cells):

        partition, world, dox_w, is_declarative = Verb.initialize(num_worlds, max_cells)

        partition = fill_in_partition(partition, num_worlds)
        which_cell = np.random.randint(len(partition))
        cell = partition[which_cell]
        # add at least 1 element of cell to dox_w
        how_many = 1 + np.random.randint(len(cell))
        dox_w[np.random.choice(cell, [how_many], replace=False)] = 1

        # TODO: was this a bug...?
        if is_declarative:
            partition = (partition[which_cell],)

        return partition, world, dox_w

    @staticmethod
    def verify_true(partition, world, dox_w, is_declarative):

        return num_cells_intersect(partition, dox_w) == 1


class Knopinion(Verb):
    """Verb meaning: \P \w: w in dox_w and (dox_w in P or dox_w in neg-P)
    """

    @staticmethod
    def generate_true(num_worlds, max_cells):

        partition, world, dox_w, _ = Verb.initialize(num_worlds, max_cells)

        world_cell = cell_of_elt(partition, world, num_worlds)

        # add at least 1 element of cell to dox_w
        how_many = 1 + np.random.randint(len(world_cell))
        dox_w[np.random.choice(world_cell, [how_many], replace=False)] = 1
        dox_w[world] = 1

        return partition, world, dox_w

    @staticmethod
    def verify_true(partition, world, dox_w, is_declarative):

        world_cell = cell_of_elt(partition, world, len(dox_w))
        dox_cell = np.nonzero(dox_w)[0]
        dox_sub_w = list_subset(dox_cell, world_cell)

        w_in_dox_w = dox_w[world] == 1

        return dox_sub_w and w_in_dox_w


class Wondows(Verb):
    """Verb meaning: \P \w: w in info(P) and dox_w subset info(P) and
    for every q in alt(f), dox_w \cap q is not empty
    """

    @staticmethod
    def generate_true(num_worlds, max_cells):

        partition, world, dox_w, is_declarative = Verb.initialize(num_worlds, max_cells)

        if is_declarative and not in_partition(partition, world):
            # proposition has to be true at w!
            partition = ((partition[0] + (world,)),)

        for cell in partition:
            how_many = 1 + np.random.randint(len(cell))
            dox_w[np.random.choice(cell, [how_many], replace=False)] = 1

        # dox_w[world] = 1

        return partition, world, dox_w

    @staticmethod
    def verify_true(partition, world, dox_w, is_declarative):

        in_info_q = in_partition(partition, world)
        intersect_every_cell = num_cells_intersect(partition, dox_w) == len(partition)

        return in_info_q and intersect_every_cell


class WondowLess(Verb):
    """Verb meaning: \P \w: dox_w subset info(P) and
    for every q in alt(f), dox_w \cap q is not empty
    """

    @staticmethod
    def generate_true(num_worlds, max_cells):

        partition, world, dox_w, _ = Verb.initialize(num_worlds, max_cells)

        for cell in partition:
            how_many = 1 + np.random.randint(len(cell))
            dox_w[np.random.choice(cell, [how_many], replace=False)] = 1

        return partition, world, dox_w

    @staticmethod
    def verify_true(partition, world, dox_w, is_declarative):

        in_info_q = in_partition(partition, world)
        intersect_every_cell = num_cells_intersect(partition, dox_w) == len(partition)

        return in_info_q and intersect_every_cell


class AllOpen(Verb):
    """Verb meaning: \P \w:
    for every q in alt(f), dox_w \cap q is not empty
    """

    @staticmethod
    def generate_true(num_worlds, max_cells):

        partition, world, dox_w, is_declarative = Verb.initialize(num_worlds, max_cells)

        for cell in partition:
            how_many = 1 + np.random.randint(len(cell))
            dox_w[np.random.choice(cell, [how_many], replace=False)] = 1

        return partition, world, dox_w

    @staticmethod
    def verify_true(partition, world, dox_w, is_declarative):

        intersect_every_cell = num_cells_intersect(partition, dox_w) == len(partition)

        return intersect_every_cell


class BelieveInfo(Verb):
    """Verb meaning: \P \w: dox_w subset info(P)
    """

    @staticmethod
    def generate_true(num_worlds, max_cells):

        partition, world, dox_w, is_declarative = Verb.initialize(num_worlds, max_cells)

        flattened = flatten_partition(partition)

        how_many = 1 + np.random.randint(len(flattened))
        dox_w[np.random.choice(flattened, [how_many], replace=False)] = 1

        return partition, world, dox_w

    @staticmethod
    def verify_true(partition, world, dox_w, is_declarative):

        flattened = np.array(flatten_partition(partition))
        flat_as_array = np.zeros(len(dox_w)).astype(int)
        flat_as_array[flattened] = 1

        return (dox_w & flat_as_array == dox_w).all()


class BelPart(Verb):
    """Verb meaning: \P \w: exists X \subseteq alt(P) s.t. \cup X \neq W and dox_w subset \cup X
    """

    @staticmethod
    def generate_true(num_worlds, max_cells):

        partition, world, dox_w, is_declarative = Verb.initialize(num_worlds, max_cells)

        # for an n-cell partition, select up to n-1 cells
        # max(1, ...) for the declarative case, where there's only one cell
        num_cells = 1 + np.random.randint(max(1, len(partition)-1))
        cells = random.sample(partition, num_cells)

        for cell in cells:
            how_many = 1 + np.random.randint(len(cell))
            dox_w[np.random.choice(cell, [how_many], replace=False)] = 1

        return partition, world, dox_w

    @staticmethod
    def verify_true(partition, world, dox_w, is_declarative):

        """
        if is_declarative:
            dox_cell = np.nonzero(dox_w)[0]
            # whether dox_w is a subset of P
            dox_sub_X = list_subset(dox_cell, partition[0])
            if not dox_sub_X:
                return False
        """

        num_worlds = len(dox_w)
        # add complement if declarative
        partition = fill_in_partition(partition, num_worlds)
        # get cells that dox_w is a subset of
        intersecting = intersecting_cells(partition, dox_w)
        # get all worlds in those cells
        flattened = flatten_partition(intersecting)

        # make sure it's not all worlds
        return len(flattened) != num_worlds

class AlmostBel(Verb):
    """Verb meaning: \P \w: exists X subset alt(P) w/ |X| <= 2 s.t. dox_w subset \cup X
    """

    @staticmethod
    def generate_true(num_worlds, max_cells):

        partition, world, dox_w, is_declarative = Verb.initialize(num_worlds, max_cells)

        # for an n-cell partition, select up to n-1 cells
        # max(1, ...) for the declarative case, where there's only one cell
        cells = random.sample(partition, max(1, 2 if len(partition) > 1 else 1))

        for cell in cells:
            how_many = 1 + np.random.randint(len(cell))
            dox_w[np.random.choice(cell, [how_many], replace=False)] = 1

        return partition, world, dox_w

    @staticmethod
    def verify_true(partition, world, dox_w, is_declarative):

        dox_cell = np.nonzero(dox_w)[0]

        if is_declarative:
            return list_subset(dox_cell, partition[0])

        intersecting = intersecting_cells(partition, dox_w)
        return len(intersecting) <= 2 and list_subset(dox_cell, flatten_partition(intersecting))



def get_all_verbs():
    return globals()["Verb"].__subclasses__()