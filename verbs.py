# TODO: document and license!
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


def partition_from_embedding(embedding):
    # get unique rows of the partition, along with array of indices that could
    # reconstruct the embedding
    unique, indices = np.unique(embedding, return_inverse=True, axis=0)
    # partition got by 'summing' the rows, since they have no non-zero
    # components in common
    # add 1 to indices to conform to convention about declarative vs question
    return np.sum(unique*(indices+1), axis=0)


def generate_partition(num_worlds, is_declarative):
    upper_bound = 2 if is_declarative else num_worlds
    partition = np.random.choice(np.arange(upper_bound), size=[num_worlds])
    # convention: values are 1...N for inquisitive meanings, 0/1 for
    # declarative meanings; so we shift the whole partition up 1 if this is not
    # a declarative
    if not is_declarative:
        partition += 1
    return partition


class Verb(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    @classmethod
    def name(cls):
        pass

    @abc.abstractmethod
    def generate_true(num_worlds):
        pass

    @abc.abstractmethod
    def generate_false(num_worlds):
        pass

    @classmethod
    def generate(cls, num_worlds, truth_value):
        return (cls.generate_true(num_worlds) if truth_value else
                cls.generate_false(num_worlds))


class Know(Verb):

    __name__ = 'know'

    @property
    @classmethod
    def name(cls):
        return cls.__name__

    @staticmethod
    def generate_true(num_worlds):

        is_declarative = np.random.random() < 0.5
        partition = generate_partition(num_worlds, is_declarative)
        world = np.random.randint(num_worlds)

        if is_declarative:
            # proposition has to be true at w!
            partition[world] = 1

        dox_w = np.zeros([num_worlds])
        # get the true answer at w
        world_cell = np.where(partition == partition[world])[0]
        # randomly include worlds from Q_w
        to_add = world_cell[np.random.random(len(world_cell)) < 0.5]
        dox_w[to_add] = 1
        # but w has to be in dox_w
        dox_w[world] = 1
        return partition, world, dox_w

    @staticmethod
    def generate_false(num_worlds):

        is_declarative = np.random.random() < 0.5
        partition = generate_partition(num_worlds, is_declarative)
        world = np.random.randint(num_worlds)

        # two ways of being false: w \notin dox_w, or dox_w is not a subset of
        # Q_w.  for now, these are weighted equally, but should they be?
        # also, both are possible, even with the current coin flip, which seems
        # like a good thing
        dox_w = np.random.choice([0.0, 1.0], size=[num_worlds])
        not_world_cell = np.where(partition != partition[world])[0]
        if np.random.random() < 0.5 or len(not_world_cell) == 0:
            # make w not in dox_w
            dox_w[world] = 0
        else:
            # make dox_w not a subset of Q_w; only possible if Q_w != W, whence
            # the check on length in the if clause
            how_many_outside = np.random.randint(len(not_world_cell))
            to_include = np.random.choice(not_world_cell,
                                          size=how_many_outside+1,
                                          replace=False)
            dox_w[to_include] = 1
        return partition, world, dox_w


if __name__ == "__main__":

    print 'T: ' + str(Know.generate_true(5))
    print 'F: ' + str(Know.generate_false(5))
