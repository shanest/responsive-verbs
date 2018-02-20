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
    return np.sum(unique*(indices+1), axis=0)


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
        dox_w = np.zeros([num_worlds])
        return partition, world, dox_w, is_declarative


class Know(Verb):

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
        # but w has to be in dox_w
        dox_w[world] = 1
        return partition, world, dox_w

    @staticmethod
    def generate_false(num_worlds):

        partition, world, dox_w, _ = Verb.initialize(num_worlds)

        # two ways of being false: w \notin dox_w, or dox_w is not a subset of
        # Q_w.  for now, these are weighted equally, but should they be?
        # also, both are possible, even with the current coin flip, which seems
        # like a good thing
        dox_w[np.random.random(len(dox_w)) < 0.5] = 1
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


class Guess(Verb):

    @staticmethod
    def generate_true(num_worlds):

        partition, world, dox_w, _ = Verb.initialize(num_worlds)

        # choose which cell
        which_cell = np.random.choice(np.unique(partition))
        cell = np.where(partition == which_cell)[0]
        to_add = cell[np.random.random(len(cell)) < 0.5]
        dox_w[to_add] = 1

        # make sure dox_w is not empty!
        if np.sum(dox_w) == 0:
            dox_w[np.random.randint(num_worlds)] = 1

        return partition, world, dox_w

    @staticmethod
    def generate_false(num_worlds):

        partition, world, dox_w, is_declarative = Verb.initialize(num_worlds)

        while len(np.unique(partition)) == 1:
            # impossible for Guess to be false of a single-cell partition, so
            # re-generate until it's not
            partition = generate_partition(num_worlds, is_declarative)

        unique = np.unique(partition)
        how_many = max(2, np.random.randint(len(unique)) + 1)
        # get at least two different cells; note replace=False!
        cell_values = np.random.choice(unique, [how_many], replace=False)
        for idx in range(how_many):
            cell = np.where(partition == cell_values[idx])[0]
            to_add = cell[np.random.random(len(cell)) < 0.5]
            dox_w[to_add] = 1
            # make sure one world from each cell gets added
            if len(to_add) == 0:
                dox_w[np.random.choice(cell)] = 1

        return partition, world, dox_w


class Wondows(Verb):

    @staticmethod
    def generate_true(num_worlds):

        partition, world, dox_w, is_declarative = Verb.initialize(num_worlds)

        # need P to be true at w for declarative
        if is_declarative:
            partition[world] = 1

        # 1. dox_w is a subset of info(Q)
        info_q = np.nonzero(partition)[0]
        to_add = info_q[np.random.random(len(info_q)) < 0.5]
        dox_w[to_add] = 1
        if len(to_add) == 0:
            dox_w[np.random.choice(info_q)] = 1

        # 2. w in dox_w
        dox_w[world] = 1

        # 3. non-empty intersection with every q in alt(Q)
        cells = np.unique(partition)
        alternatives = cells[np.nonzero(cells)]
        for idx in range(len(alternatives)):
            # TODO: refactor this, and its use in Guess.generate_false, out
            cell = np.where(partition == alternatives[idx])[0]
            to_add = cell[np.random.random(len(cell)) < 0.5]
            dox_w[to_add] = 1
            # make sure one world from each cell gets added
            if len(to_add) == 0:
                dox_w[np.random.choice(cell)] = 1

        return partition, world, dox_w

    @staticmethod
    def generate_false(num_worlds):

        partition, world, dox_w, _ = Verb.initialize(num_worlds)

        dox_w[np.random.random(len(dox_w)) < 0.5] = 1

        # if info_q != W, 50% chance of adding non-info worlds in
        not_info_q = np.where(partition == 0)[0]
        if len(not_info_q) > 0 and np.random.random() < 0.5:
            how_many = max(1, np.random.randint(len(not_info_q)))
            to_add = np.random.choice(not_info_q, [how_many], replace=False)
            dox_w[to_add] = 1

        # or make w not in dox_w
        if np.random.random() < 0.5:
            dox_w[world] = 0
        # or make empty intersection with some alternative
        else:
            random_cell = np.where(partition ==
                                   partition[np.random.choice(
                                       np.unique(partition))])[0]
            dox_w[random_cell] = 0

        if not np.any(dox_w):
            # make sure dox_w is not empty
            dox_w[np.random.randint(num_worlds)] = 1

        return partition, world, dox_w


class Knopinion(Verb):

    @staticmethod
    def generate_true(num_worlds):

        partition, world, dox_w, _ = Verb.initialize(num_worlds)

        not_info_q = np.where(partition == 0)[0]
        if len(not_info_q) > 0:
            # opinionated: dox_w subset of one of P or ~P
            cell = np.where(partition ==
                            partition[np.random.choice(np.unique(partition))])[0]
            to_add = cell[np.random.random(len(cell)) < 0.5]
            dox_w[to_add] = 1
            if not np.any(dox_w):
                # make sure dox_w is not empty
                dox_w[np.random.choice(cell)] = 1
        else:
            # dox_w subset of Q_w
            world_cell = np.where(partition == partition[world])[0]
            to_add = world_cell[np.random.random(len(world_cell)) < 0.5]
            dox_w[to_add] = 1
            if not np.any(dox_w):
                # make sure dox_w is not empty
                dox_w[np.random.choice(world_cell)] = 1

        return partition, world, dox_w

    @staticmethod
    def generate_false(num_worlds):

        partition, world, dox_w, is_declarative = Verb.initialize(num_worlds)

        while len(np.unique(partition)) == 1:
            # impossible for Knopinion to be false of a single-cell partition,
            # so re-generate until it's not
            partition = generate_partition(num_worlds, is_declarative)

        # add some not Q_w worlds to dox_w
        not_world_cell = np.where(partition != partition[world])[0]
        how_many = max(1, np.random.randint(len(not_world_cell)))
        dox_w[np.random.choice(not_world_cell, [how_many], replace=False)] = 1

        world_cell = np.where(partition == partition[world])[0]
        how_many = max(1, np.random.randint(len(world_cell)))
        if is_declarative:
            # declarative has to have some world_cell elements
            how_many = max(1, how_many)
        dox_w[np.random.choice(world_cell, [how_many], replace=False)] = 1

        return partition, world, dox_w


if __name__ == "__main__":

    print 'T: ' + str(Know.generate_true(5))
    print 'F: ' + str(Know.generate_false(5))
