from collections import deque


class LevelBuffer:
    def __init__(self, size: int):
        self.size = size
        self.buffer = deque(maxlen=size)

    def add(self, value):
        self.buffer.append(value)

    def get(self):
        return list(self.buffer)
