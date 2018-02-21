import numpy as np
import verbs


class DataGenerator(object):

    def __init__(self, verbs, num_worlds, items_per_bin=1000,
                 max_tries_per_bin=5000, test_bin_size=200):
        self.verbs = verbs
        self.num_worlds = num_worlds
        self.items_per_bin = items_per_bin
        self.max_tries_per_bin = max_tries_per_bin
        self.one_hots = np.eye(num_worlds)
        self.verb_labels = np.eye(len(verbs))
        self.point_length = num_worlds**2 + 2*num_worlds
        self.base_seq = 2**np.arange(self.point_length)
        self.test_bin_size = test_bin_size

        self.data = {(verb, truth_value): [] for verb in verbs for
                     truth_value in (True, False)}
        self.generate_data()

    def generate_data(self):

        for (verb, truth_value) in self.data:
            generated = set()
            tries = 0
            while (len(generated) < self.items_per_bin and
                   tries < self.max_tries_per_bin):
                # get data point from verb
                partition, world, v_w = verb.generate(self.num_worlds,
                                                      truth_value)
                # convert point to binary vector
                point = self.generate_point(partition, world, v_w)
                # make point hashable, so can be added to set
                hashable = self.point_to_int(point)
                if hashable not in generated:
                    self.data[(verb, truth_value)].append(point)
                    generated.add(hashable)
                tries += 1

        # TODO: balance by over/under-sampling here? Or will it not be needed?
        print 'Generated this many data points:'
        print {k: len(self.data[k]) for k in self.data}
        self.test_bins = {k: self.data[k][:self.test_bin_size]
                          for k in self.data}
        self.train_bins = {k: self.data[k][self.test_bin_size:]
                           for k in self.data}

    def generate_point(self, partition, world, v_w):
        embedding = verbs.embedding(partition)
        embedding_vec = np.reshape(embedding, -1)
        world_vec = self.one_hots[world]
        return np.concatenate((embedding_vec, world_vec, v_w))

    def point_to_int(self, point):
        return np.sum(self.base_seq * point)

    def get_training_data(self, shuffle=True):
        return self._prep_dataset(self.train_bins, shuffle)

    def get_test_data(self, shuffle=False):
        return self._prep_dataset(self.test_bins, shuffle)

    def _prep_dataset(self, data, shuffle=True):
        all_data = []

        for (verb, truth_value) in data:
            verb_label = self.verb_labels[self.verbs.index(verb)]
            all_data.extend([(np.concatenate([point, verb_label]),
                              # note: TF just wants label as integer
                              int(truth_value))
                             for point in data[(verb, truth_value)]])

        if shuffle:
            np.random.shuffle(all_data)

        Xs = np.array([datum[0] for datum in all_data])
        Ys = np.array([datum[1] for datum in all_data])
        return Xs, Ys
