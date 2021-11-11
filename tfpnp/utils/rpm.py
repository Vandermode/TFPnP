import random


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.index = 0

    def store(self, obj):
        if self.size() > self.capacity:
            print('buffer size larger than capacity, trimming...')
            self.buffer = self.buffer[(self.size() - self.capacity):]
        elif self.size() == self.capacity:
            self.buffer[self.index] = obj
            self.index += 1
            self.index %= self.capacity
        else:
            self.buffer.append(obj)

    def size(self):
        return len(self.buffer)

    def sample_batch(self, env_batch):
        if self.size() < env_batch:
            index_value = random.sample(
                list(enumerate(self.buffer)), self.size())
        else:
            index_value = random.sample(
                list(enumerate(self.buffer)), env_batch)
        indexes = []
        values = []
        for idx, val in index_value:
            indexes.append(idx)
            values.append(val)
        return values


class GroupReplayMemory:
    def __init__(self, capacity, keys):
        self.capacity = capacity
        self.keys = keys
        self.buffer = {}
        self.index = {}
        for key in keys:
            self.buffer[key] = []
            self.index[key] = 0
    
    def key_from_ob(self, obj):
        raise NotImplementedError
    
    def store(self, obj):
        key = self.key_from_ob(obj)
        if self._size(key) > self.capacity:
            print('buffer size larger than set value, trimming...')
            self.buffer = self.buffer[key][(self._size(key) - self.capacity):]
        elif self._size(key) == self.capacity:
            self.buffer[key][self.index[key]] = obj
            self.index[key] += 1
            self.index[key] %= self.capacity
        else:
            self.buffer[key].append(obj)

    def _size(self, key):
        return len(self.buffer[key])
    
    def size(self):
        return sum([len(self.buffer[key]) for key in self.keys])

    def sample_batch(self, env_batch):
        keys = [key for key in self.keys if self._size(key) > 0]
        key = random.choice(keys)
        size = self._size(key)

        if size < env_batch:
            index_value = random.sample(
                list(enumerate(self.buffer[key])), size)
        else:
            index_value = random.sample(
                list(enumerate(self.buffer[key])), env_batch)
        indexes = []
        values = []            
        for idx, val in index_value:
            indexes.append(idx)
            values.append(val)
        return values
