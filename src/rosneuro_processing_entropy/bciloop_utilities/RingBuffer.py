class RingBuffer:
    """ class that implements a not-yet-full buffer """
    def __init__(self, size_max):
        self.max = size_max
        self.data = []
        self.cur = 0
        self.isFull = False

    class __Full:
        """ class that implements a full buffer """
        def append(self, x):
            """ Append an element overwriting the oldest one. """
            self.data = self.data[len(x):]
            self.data.extend(x)

        def get(self):
            """ return list of elements in correct order """
            return self.data

    def append(self, x):
        """append an element at the end of the buffer"""
        self.data.extend(x)
        if len(self.data) >= self.max:
            # Permanently change self's class from non-full to full
            self.data = self.data[len(self.data)-self.max:]
            self.__class__ = self.__Full
            self.isFull = True

    def get(self):
        return self.data


# sample usage
# if __name__=='__main__':
#     x=RingBuffer(6)
#     x.append([1, 2])
#     print(x.__class__, x.get())
#     x.append([3, 4])
#     print(x.__class__, x.get())
#     x.append([5, 6])
#     print(x.__class__, x.get())
#     x.append([7, 8])
#     print(x.__class__, x.get())
#     x.append([9, 10])
#     print(x.__class__, x.get())
#     x.append([11, 12])
#     print(x.__class__, x.get())
#     x.append([13, 14])
#     print(x.__class__, x.get())