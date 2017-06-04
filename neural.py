import random
import math
import json

# neural network class with a sigmoid squashing function

class NeuralNetwork:

    def __init__(self, loadfile=None, num_inputs=0, num_hidden=0, num_outputs=0, learning_rate = 0.01, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):

        # load parameters and weights from json file
        if loadfile:
            with open(loadfile) as infile:
                data = json.load(infile)
                hidden_layer_weights = data['layers']['hidden']
                output_layer_weights = data['layers']['output']
                learning_rate = data['learning_rate']
                num_inputs = data['num_inputs']
                output_layer_bias = data['output_layer_bias']
                hidden_layer_bias = data['hidden_layer_bias']
                num_outputs = len(data['layers']['output'])
                num_hidden = len(data['layers']['hidden'])
        
        self.output_layer_bias = output_layer_bias
        self.hidden_layer_bias = hidden_layer_bias

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        # initialise layers and their neurons
        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)

        self.learning_rate = learning_rate

        random.seed(0) # seed random numbers

        # initialise hidden / output layer weights a random number or supplied weights
        self.init_hidden_layer_weights(hidden_layer_weights)
        self.init_output_layer_weights(output_layer_weights)

    def init_hidden_layer_weights(self, hidden_layer_weights):
        for h in range(len(self.hidden_layer.neurons)):
            if hidden_layer_weights:
                self.hidden_layer.neurons[h].weights = hidden_layer_weights[h]
            else:
                for i in range(self.num_inputs):
                    self.hidden_layer.neurons[h].weights.append(random.random())

    def init_output_layer_weights(self, output_layer_weights):
        for o in range(len(self.output_layer.neurons)):
            if output_layer_weights:
                self.output_layer.neurons[o].weights = output_layer_weights[o]
            else:
                for h in range(len(self.hidden_layer.neurons)):
                    self.output_layer.neurons[o].weights.append(random.random())

    def save(self,filename):

        with open(filename,"w") as outfile:

            data = {
                'layers':{
                    'hidden':[],
                    'output':[]
                }
            }

            for h in range(len(self.hidden_layer.neurons)):
                data['layers']['hidden'].append(self.hidden_layer.neurons[h].weights)

            for o in range(len(self.output_layer.neurons)):
                data['layers']['output'].append(self.output_layer.neurons[o].weights)

            data['learning_rate'] = self.learning_rate
            data['hidden_layer_bias'] = self.hidden_layer_bias
            data['output_layer_bias'] = self.output_layer_bias
            data['num_inputs'] = self.num_inputs

            json.dump(data,outfile)

    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    # given an input and target vector, adjust weights to minimise the error
    def backpropagate(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        # output neuron delta
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].error_wrt_total_input(training_outputs[o])

        # hiddne neuron delta
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):

            # derivative of error wrt output of each hidden layer neuron
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]

            # sum of error for hidden neuron
            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].total_input_wrt_input()

        # update output neuron weights to minimise error
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                # partial derivatives multiplied
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].total_input_wrt_weight(w_ho)
                # update weight w learning rate
                self.output_layer.neurons[o].weights[w_ho] -= self.learning_rate * pd_error_wrt_weight

        # update hidden weights to minmise error
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):
                # partial derivatives multiplied
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[h].total_input_wrt_weight(w_ih)
                # update weight w learning rate
                self.hidden_layer.neurons[h].weights[w_ih] -= self.learning_rate * pd_error_wrt_weight

    # sum of MSE for each training set
    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error

# neuron layer subclass for feeding input vectors (from input to hidden layer or hidden to output layer)
# and collecting outputs for each neuron in the layer
class NeuronLayer:
    def __init__(self, num_neurons, bias):

        if bias: self.bias = bias 
        else: bias = random.random() # random bias if none is present

        self.neurons = []
        for i in range(num_neurons): self.neurons.append(Neuron(self.bias))

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons: outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons: outputs.append(neuron.output)
        return outputs

# neuron subclass for feeding input vectors (from input to hidden layer or hidden to output layer)
# and returning outputs
class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    # get output from squashing function
    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.sigmoid(self.total_net_input())
        return self.output

    # net input (inputs * weights)
    def total_net_input(self):
        total = 0
        for i in range(len(self.inputs)): total += self.inputs[i] * self.weights[i]
        return total + self.bias

    # sigmoid squashing function
    def sigmoid(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    # partial derivative of input wrt total net input (delta)
    def error_wrt_total_input(self, target_output):
        return self.error_wrt_output(target_output) * self.total_input_wrt_input();

    # mean square error
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    # partial derivative of error wrt target output
    def error_wrt_output(self, target_output):
        return -(target_output - self.output)

    # partial derivative wrt visible input (logistic function)
    def total_input_wrt_input(self):
        return self.output * (1 - self.output)

    # partial derivative of error wrt weight
    def total_input_wrt_weight(self, index):
        return self.inputs[index]
