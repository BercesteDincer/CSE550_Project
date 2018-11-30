#Parent class for all messages in the system
class Message:
    def __init__(self, source):
        self.source = source #pid of the sender of the message

#Message class for sending the request of the client to the PROPOSER
class TrainingMsg(Message):
    def __init__(self, source, start, end):
        super(TrainingMsg, self).__init__(source)
        self.start = start
        self.end = end
    def __str__(self):
        return "TRAIN MESSAGE = [Starting = {}, Ending = {}]".format(self.start, self.end)

