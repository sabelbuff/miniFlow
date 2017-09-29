import numpy as np


class Node(object):
    def __init__(self, inbound_nodes=[]):
        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = []

        self.value = []

        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

    def forward(self):
        raise NotImplemented


class Input(Node):
    def __int__(self):
        Node.__init__(self)

    def forward(self, value=None):
        if value is not None:
            self.value = value


class Add(Node):
    def __int__(self, *inputs):
        Node.__init__(self, [inputs])

    def forward(self):
        temp_sum = 0
        for n in self.inbound_nodes:
            temp_sum += n.value
        self.value = temp_sum


class Mul(Node):
    def __int__(self, *inputs):
        Node.__init__(self, [inputs])

    def forward(self):
        temp_prod = 1
        for n in self.inbound_nodes:
            temp_prod += n.value
        self.value = temp_prod


class Linear(Node):
    def __int__(self, X, W, b):
        Node.__init__(self, [X, W, b])

    def forward(self):

        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value

        self.value = np.dot(X, W) + b


class Sigmoid(Node):
    def __int__(self, x):
        Node.__init__(self, [x])

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        input_value = self.inbound_nodes[0].value
        self.value = self._sigmoid(input_value)


class MSE(Node):
    def __int__(self, y, a):
        Node.__init__(self, [y, a])

    def forward(self):
        output_value = self.inbound_nodes[0].value.reshape(-1, 1)
        target = self.inbound_nodes[1].value.reshape(-1, 1)
        self.value = np.mean(np.square(target - output_value))





