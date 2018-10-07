import numpy as np
import math

if __name__ == "__main__":
    data_set = []

    D = [0.244, 0.294, 0.351, 0.381, 0.420, 0.459, 0.518, 0.574, 0.600, 0.636, 0.648, 0.661, 0.681, 0.708, 0.746]

    EntD = 0
    for item in D:
        EntD += -item*math.log(item)