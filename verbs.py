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
import numpy as np


def embedding(partition):
    """This function implements the E function from Theiler, Aloni, and
    Roelofsen.  In particular, it returns a square matrix of size num_worlds,
    with each row of the matrix corresponding to the complete answer to the
    question represented by partition in that world.

    See module doc-string for notational conventions used throughout.

    Args:
        partition: a 1-d vector, should be an ndarray

    Returns:
        matrix of shape (len(partition), len(parttition))
    """
    num_worlds = len(partition)
    result = np.zeros((num_worlds, num_worlds))
    for world in range(num_worlds):
        # this treats 0 as special, not part of a partition, so only used in
        # declarative meanings
        if partition[world] != 0:
            cell = np.where(partition == partition[world])
            result[world, cell] = 1
    return result


def partition_as_matrix(partition):
    """Convert a partition to a matrix.

    Args:
        partition: 1-D numpy array

    Returns:
        a [len(partition), len(partition)] size matrix, where rows are one-hot
        vectors corresponding to the elements of partition, except that 0 gets
        mapped to the all-zeros vector
    """
    cells = np.vstack((np.zeros(len(partition)),
                       np.eye(len(partition))))
    return np.array([cells[c] for c in np.nditer(partition)])


def partition_from_embedding(embedding):
    """Returns a partition from its embedding.  While embedding() is a
    many->one function, this is one->one.  So, one way of checking whether two
    partitions are equal is to check whether
    partition_from_embedding(embedding(partition1)) ==
    partition_from_embedding(embedding(partition2)).

    Args:
        embedding: an NxN binary matrix, assumed to be generated by embedding()

    Returns:
        a 1-D numpy array, the partition that generated the embedding
    """
    # get unique rows of the partition, along with array of indices that could
    # reconstruct the embedding
    unique, indices = np.unique(embedding, return_inverse=True, axis=0)
    # partition got by 'summing' the rows, since they have no non-zero
    # components in common
    # add 1 to indices to conform to convention about declarative vs question
    return np.sum(unique*(indices+1), axis=0, dtype=np.int_)


def generate_partition(num_worlds, is_declarative):
    """Generates a partition, which is just a 1-D array of integers.

    Args:
        num_worlds: plays two roles. (i) length of partition, (ii) maximum
                    number of possible cells
        is_declarative: boolean

    Returns:
        1-D array of length num_worlds.
        If is_declarative, it is a binary array of 0s and 1s.
        If not is_declarative, the minimum value is 1; there are no 0s.
    """
    # TODO: forbid single-cell partitions? or, since they're 1/2^N chance of
    # being generated, just not worry about it?
    # geometric distribution biases towards small number of cells; play with
    # parameter 0.2?
    upper_bound = 2 if is_declarative else min(num_worlds,
                                               1+np.random.geometric(0.2))
    partition = np.random.choice(np.arange(upper_bound), size=[num_worlds])
    # convention: values are 1...N for inquisitive meanings, 0/1 for
    # declarative meanings; so we shift the whole partition up 1 if this is not
    # a declarative
    if not is_declarative:
        partition += 1
    return partition
# TODO: refactor add_from( )


class Verb(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def generate_true(num_worlds):
        """Generate true examples of your Verb.

        Args:
            num_worlds: how many worlds

        Returns:
            partition: 1-D array, a partition
            world: int, the actual world
            dox_w: 1-D array of 1s and 0s, the agent's doxastic state
        """
        pass

    @abc.abstractmethod
    def generate_false(num_worlds):
        """Generate false examples of your Verb.

        Args:
            num_worlds: how many worlds

        Returns:
            partition: 1-D array, a partition
            world: int, the actual world
            dox_w: 1-D array of 1s and 0s, the agent's doxastic state
        """
        pass

    @classmethod
    def generate(cls, num_worlds, truth_value):
        """Generate examples.  Calls generate_true or generate_false based on
        the argument truth_value.
        """
        return (cls.generate_true(num_worlds) if truth_value else
                cls.generate_false(num_worlds))

    @staticmethod
    def initialize(num_worlds):
        """Perform initial prep for generate functions.

        Args:
            num_worlds: how many worlds

        Returns:
            partition: generate_partition(num_worlds, is_declarative)
            world: a random integer up to num_worlds
            dox_w: a 1-D zero array of length num_worlds
            is_declarative: boolean, whether it's a declarative example or not
        """
        is_declarative = np.random.random() < 0.5
        partition = generate_partition(num_worlds, is_declarative)
        world = np.random.randint(num_worlds)
        dox_w = np.zeros([num_worlds], dtype=np.int_)
        return partition, world, dox_w, is_declarative


class Know(Verb):
    """Verb meaning: \Q \w: dox_w is a subset of Q_w and f(w) != empty
    """

    @staticmethod
    def generate_true(num_worlds):

        partition, world, dox_w, is_declarative = Verb.initialize(num_worlds)

        if is_declarative:
            # proposition has to be true at w!
            partition[world] = 1

        # get the true answer at w
        world_cell = np.where(partition == partition[world])[0]
        # randomly include worlds from Q_w
        to_add = world_cell[np.random.random(len(world_cell)) < 0.5]
        dox_w[to_add] = 1

        # make sure dox_w not empty
        if np.sum(dox_w) == 0:
            dox_w[np.random.choice(world_cell)] = 1

        return partition, world, dox_w

    @staticmethod
    def generate_false(num_worlds):

        partition, world, dox_w, is_declarative = Verb.initialize(num_worlds)

        dox_w[np.random.random(len(dox_w)) < 0.5] = 1

        not_Qw = np.where(partition != partition[world])[0]
        while len(not_Qw) == 0:
            partition = generate_partition(num_worlds, is_declarative)
            not_Qw = np.where(partition != partition[world])[0]

        if np.sum(dox_w[not_Qw]) == 0:
            # get at least one not_Qw world
            how_many = 1 + np.random.randint(len(not_Qw))
            to_add = np.random.choice(not_Qw, [how_many], replace=False)
            dox_w[to_add] = 1

        return partition, world, dox_w


class BeCertain(Verb):
    """Verb meaning: \Q \w: dox_w is a subset of Q_w for some w
    """

    @staticmethod
    def generate_true(num_worlds):

        partition, world, dox_w, is_declarative = Verb.initialize(num_worlds)

        cell = np.where(partition == np.random.choice(np.unique(partition)))[0]
        # add at least 1 element of cell to dox_w
        how_many = 1 + np.random.randint(len(cell))
        dox_w[np.random.choice(cell, [how_many], replace=False)] = 1

        return partition, world, dox_w

    @staticmethod
    def generate_false(num_worlds):

        partition, world, dox_w, is_declarative = Verb.initialize(num_worlds)

        while len(np.unique(partition)) == 1:
            # impossible for Opiknow to be false of a single-cell partition,
            # so re-generate until it's not
            partition = generate_partition(num_worlds, is_declarative)

        while len(np.unique(partition[np.nonzero(dox_w)[0]])) < 2:
            dox_w = np.random.choice([0, 1], [num_worlds])

        return partition, world, dox_w


class Knopinion(Verb):
    """Verb meaning: \Q \w: dox_w is a subset of Q_w or dox_w is in inq-neg(Q)
    """

    @staticmethod
    def generate_true(num_worlds):

        partition, world, dox_w, _ = Verb.initialize(num_worlds)

        not_info_q = np.where(partition == 0)[0]
        cell_value = (partition[world] if len(not_info_q) == 0
                      else np.random.choice(np.unique(partition)))

        cell = np.where(partition == cell_value)[0]
        # add at least 1 element of cell to dox_w
        how_many = 1 + np.random.randint(len(cell))
        dox_w[np.random.choice(cell, [how_many], replace=False)] = 1

        return partition, world, dox_w

    @staticmethod
    def generate_false(num_worlds):

        partition, world, dox_w, is_declarative = Verb.initialize(num_worlds)

        dox_w[np.random.random(len(dox_w)) < 0.5] = 1

        while len(np.unique(partition)) == 1:
            # impossible for Knopinion to be false of a single-cell partition,
            # so re-generate until it's not
            partition = generate_partition(num_worlds, is_declarative)

        # add some not Q_w worlds to dox_w
        not_world_cell = np.where(partition != partition[world])[0]
        if np.sum(dox_w[not_world_cell]) == 0:
            how_many = 1 + np.random.randint(len(not_world_cell))
            dox_w[np.random.choice(not_world_cell, [how_many], replace=False)] = 1

        world_cell = np.where(partition == partition[world])[0]
        if is_declarative and np.sum(dox_w[world_cell]) == 0:
            # declarative has to have some world_cell elements
            how_many = 1 + np.random.randint(len(world_cell))
            dox_w[np.random.choice(world_cell, [how_many], replace=False)] = 1

        return partition, world, dox_w


class Opiknow(Verb):
    """Verb meaning: \Q \w: f(w) != empty and dox_w is a subset of p for some p
    in alt(Q)
    """

    @staticmethod
    def generate_true(num_worlds):

        partition, world, dox_w, is_declarative = Verb.initialize(num_worlds)

        if is_declarative:
            partition[world] = 1

        cell_values = np.unique(partition[np.nonzero(partition)])
        cell = np.where(partition == np.random.choice(cell_values))[0]
        # add at least 1 element of cell to dox_w
        how_many = 1 + np.random.randint(len(cell))
        dox_w[np.random.choice(cell, [how_many], replace=False)] = 1

        return partition, world, dox_w

    @staticmethod
    def generate_false(num_worlds):

        partition, world, dox_w, is_declarative = Verb.initialize(num_worlds)

        dox_w[np.random.random(len(dox_w)) < 0.5] = 1

        if is_declarative:
            partition[world] = 1

        while len(np.unique(partition)) == 1:
            # impossible for Opiknow to be false of a single-cell partition,
            # so re-generate until it's not
            partition = generate_partition(num_worlds, is_declarative)

        while len(np.unique(partition[np.nonzero(dox_w)[0]])) < 2:
            dox_w = np.random.choice([0, 1], [num_worlds])

        return partition, world, dox_w


def get_all_verbs():
    return globals()['Verb'].__subclasses__()
