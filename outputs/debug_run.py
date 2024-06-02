from typing import *
from functools import reduce

# def sum_product(numbers: List[int]) -> Tuple[int, int]:
#     """ For a given list of integers, return a tuple consisting of a sum and a product of all the integers in a list.
#     Empty sum should be equal to 0 and empty product should be equal to 1.
#     >>> sum_product([])
#     (0, 1)
#     >>> sum_product([1, 2, 3, 4])
#     (10, 24)
#     """
#     return sum(numbers), reduce(lambda x, y: x * y, numbers, 1)

# print(sum_product([1, 2, 3, 4]))


# def fizz_buzz(n: int):
#     """Return the number of times the digit 7 appears in integers less than n which are divisible by 11 or 13.
#     >>> fizz_buzz(50)
#     0
#     >>> fizz_buzz(78)
#     2
#     >>> fizz_buzz(79)
#     3
#     """
#     count = 0
#     for i in range(1, n):
#         if i % 11 == 0 or i % 13 == 0:
#             if str(i).__contains__("7"):
#                 count += 1
#     return count

# print(fizz_buzz(78))


def sort_even(l: list):
    """This function takes a list l and returns a list l' such that
    l' is identical to l in the odd indicies, while its values at the even indicies are equal
    to the values of the even indicies of l, but sorted.
    >>> sort_even([1, 2, 3])
    [1, 2, 3]
    >>> sort_even([5, 6, 3, 4])
    [3, 6, 5, 4]
    """
    # Your code here
    even_sorted = sorted(l[::2])
    odd_sorted = l[1::2]
    return odd_sorted + even_sorted

print(sort_even([3, 6, 5, 4]))