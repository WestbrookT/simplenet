import numpy as np

np.set_printoptions(suppress=True)


def sigmoid(x, derivative=False):
    if derivative:
        return x * (1.0 - x)
    return 1.0 / (1 + np.exp(-x))


class Network:
    def __init__(self, layer_structures):

        inputCount = layer_structures[0]

        self.inputCount = inputCount
        layerCount = len(layer_structures)
        self.layers = [None] * layerCount

        lastOutput = inputCount
        for i in range(layerCount):
            outputs = layer_structures[i]

            if i == 0:
                outputs = inputCount

            layer = 2 * np.random.random((lastOutput, outputs)) - 1
            self.layers[i] = layer
            lastOutput = outputs

    def run(self, in_set, all_layers=False):

        if all_layers:
            out = [np.array(in_set)]
            last_out = np.array(in_set)

            for i, layer in enumerate(self.layers):
                last_out = sigmoid(np.dot(last_out, layer))
                out.append(last_out)

            return out
        else:
            last_out = np.array(in_set)
            for i, layer in enumerate(self.layers):
                last_out = sigmoid(np.dot(last_out, layer))

            return last_out

    def train(self, in_sets, out_sets, iterations=10):
        if len(in_sets) != len(out_sets):
            exit('TRAINING SET COUNT MISMATCH')

        for iter in range(iterations):

            out = [np.array(in_sets)]
            last_out = np.array(in_sets)

            for i, layer in enumerate(self.layers):
                last_out = sigmoid(np.dot(last_out, layer))
                out.append(last_out)

            out_layers = out

            last_error = out_sets - out_layers[-1]

            last_delta = last_error * sigmoid(out_layers[-1], True)

            deltas = [last_delta]

            for k in range(len(self.layers) - 1, 0, -1):
                layer_error = last_delta.dot(self.layers[k].T)

                layer_delta = layer_error * sigmoid(out_layers[k], True)

                last_delta = layer_delta

                deltas.append(layer_delta)

            for k in range(len(self.layers) - 1, 0, -1):
                self.layers[k] += out_layers[k].T.dot(deltas[-k - 1])

    def storage(self):
        out = []
        for layer in self.layers:
            out.append(layer.tolist())

        return out

    def load(self, layers):
        out = []
        for layer in layers:
            out.append(np.array(layer))
        self.layers = out

    def __repr__(self):
        return str(self.layers)

    def __str__(self):
        return self.__repr__()
