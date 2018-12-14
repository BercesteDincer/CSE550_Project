#Parent class for all messages in the system
class Message:
    def __init__(self, source):
        self.source = source #pid of the sender of the message

#Message class for sending the request of the client to the PROPOSER
class TrainingMsg(Message):
    def __init__(self, source, start, end, model_file, weights_file, input_file):
        super(TrainingMsg, self).__init__(source)
        self.start = start
        self.end = end
        self.model_file = model_file
        self.weights_file = weights_file
        self.input_file = input_file
    def __str__(self):
        return "TRAIN MESSAGE = [Starting = {}, Ending = {}]".format(self.start, self.end)

class ClientRequestMsg(Message):
    def __init__(self, source, model_file, weights_file, input_file):
        super(Message, self).__init__(source)
        self.model_file = model_file
        self.weights_file = weights_file
        self.input_file = input_file
    def __str__(self):
        return "CLIENT REQUEST MESSAGE = [Model file = {}, Weights File = {}, Input File = {}]".format(self.model_file, self.weights_file, self.input_file)
