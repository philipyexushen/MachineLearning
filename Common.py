import numpy as np

def lower_bound(vector, val, condition):
    left = 0
    right = len(vector)

    while left < right:
        mid = int(left + (right - left) / 2)
        if condition(vector[mid], val):
            left = mid + 1
        else:
            right = mid

    return left

def upper_bound(vector, val, condition):
    left = 0
    right = len(vector)

    while left < right:
        mid = int(left + (right - left) / 2)
        if condition(val,vector[mid]):
            right = mid
        else:
            left = mid + 1

    return left



