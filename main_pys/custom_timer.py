from collections import deque, defaultdict
import time
import numpy as np

# https://dbader.org/blog/python-context-managers-and-with-statement
# https://stackoverflow.com/questions/5109507/pass-argument-to-enter
# https://realpython.com/python-timer/


class CustomTimer:
    def __init__(self):
        self.stack = deque()
        self.time_dict = defaultdict(list)
        self.specific_calls = dict() # Used for specific start and stop calls

    def __enter__(self):
        return self

    def __call__(self, aKey):
        self.stack.append((aKey, time.time()))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if len(self.stack):
            aKey, aTime = self.stack.pop()
            dif = time.time() - aTime
            self.time_dict[aKey].append(dif)

    # Returns sum of times per key
    def getTimes(self, aKey=None, retType="sum"):
        assert(retType in ["sum", "list"])
        keys = list(self.time_dict.keys())
        if aKey is not None:
            keys = [aKey]

        retDict = dict()
        for aKey in keys:
            if retType == "list":
                retDict[aKey] = self.time_dict[aKey]
            else:
                retDict[aKey] = np.sum(self.time_dict[aKey])
        if aKey is not None:
            return retDict[aKey]
        return retDict  # Dictionary
    
    def printTimes(self, aKey=None):
        if aKey is not None:
            print(f"{aKey}: {np.sum(self.time_dict[aKey]):.3f}")
        else:
            for aKey in self.time_dict:
                print(f"{aKey}: {np.sum(self.time_dict[aKey]):.3f}")
        
    def start(self, aKey):
        self.specific_calls[aKey] = time.time()

    def stop(self, aKey):
        if aKey not in self.specific_calls.keys():
            raise KeyError(f"Key {aKey} not found in specific_calls")
        dif = time.time() - self.specific_calls[aKey]
        self.time_dict[aKey].append(dif)
        del self.specific_calls[aKey]

    # def clearKeys(self, keys=None):
    #     if keys is None:
    #         self.time_dict.clear()
    #     elif isinstance(keys, str):
    #         self.time_dict.pop(keys, None)
    #     else:
    #         for key in keys:
    #             self.time_dict.pop(key, None)

def exampleUse():
    # Example use
    t = CustomTimer()
    for i in range(10):
        with t("A1"):
            time.sleep(0.05)
        with t("B1"):
            time.sleep(1)
    values = t.getTimes()

    print(values)
    # access A1 and B1 times

if __name__ == "__main__":
    # timeDataloader()
    exampleUse()