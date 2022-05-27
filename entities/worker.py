import pickle


class Node:
    def __init__(self, data_path, model):
        with open(data_path, "rb") as file:
            data = pickle.load(file)
        self.xtrain, self.ytrain, self.xtest, self.ytest = data

    def receive(self, parameters):
        self.load_state_dict(parameters)

    def send(self):
        pass
