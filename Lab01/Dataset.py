import torch
import gzip
import pickle

class Dataset:
	
    def __init__(self, name):
        
        self.name = name
        self.testing_set = None
        self.validation_set = None
        self.training_set = None

        def _map_data(given_set):
            data = given_set[0]
            tags = given_set[1]
            output = []

            for index in range(len(tags)):
                output += [(torch.from_numpy(data[index]).view(784, 1), tags[index])]

            return output

        with gzip.open(name, "rb") as fd:
            training_set, validation_set, testing_set = pickle.load(fd, encoding='latin')

        self.training_set = _map_data(training_set)
        self.validation_set = _map_data(validation_set)
        self.testing_set = _map_data(testing_set)