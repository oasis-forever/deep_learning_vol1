class SimplePerceptron:
    def __init__(self):
        pass

    def and_gate(self, x1, x2):
        w1    = 0.5
        w2    = 0.5
        theta = 0.7
        tmp = (x1 * w1) + (x2 * w2)
        if tmp > theta:
            return 1
        else:
            return 0

    def nand_gate(self, x1, x2):
        w1    = -0.5
        w2    = -0.5
        theta = -0.7
        tmp = (x1 * w1) + (x2 * w2)
        if tmp > theta:
            return 1
        else:
            return 0

    def or_gate(self, x1, x2):
        w1    = 0.8
        w2    = 0.8
        theta = 0.7
        tmp = (x1 * w1) + (x2 * w2)
        if tmp > theta:
            return 1
        else:
            return 0
