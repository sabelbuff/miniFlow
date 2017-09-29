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
    






