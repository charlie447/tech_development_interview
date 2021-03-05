import sys
import time
import functools

def fibonacci_1(n):
    """generate Fibonacci sequence according to following function
    F(n) = 0 when n = 0
    F(n) = 1 when n = 1
    F(n) = F(n-1) + F(n-2) when n > 1

    O(2^n)
    Args:
        n (inte): the length of the Fibonacci sequence that about to generate
    """

    if n == 0:
        return 0
    
    if n == 1:
        return 1

    if n > 1:
        return fibonacci_1(n - 1) + fibonacci_1(n - 2)

def memo(func):
    cache = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

@memo
def fibonacci_4(n):
    # with caching
    if n < 2:
        return 1
    return fibonacci_4(n-1) + fibonacci_4(n-2)

def fibonacci_2(n):
    """Using generator to calculate Fibonacci list.

    Args:
        n (int): the length of the Fibonacci list.

    Yields:
        int: return a generator
    """
    a = 0
    b = 1
    counter = 0
    while True:
        if counter > n:
            return
        yield a
        a, b = b, a + b
        counter += 1


def climbStaires(n: int) -> int:
    # 滚动数组的方法
    p = 0
    q = 0
    r = 1
    if n < 2:
        return 1
    for step in range(n):
        p = q
        q = r
        r = p + q
    return r

def fibonacci_3(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return b

if __name__ == "__main__":
    # fib = fibonacci_2(10)
    start = time.time()
    # fib_1 = fibonacci_2(10)
    # steps = climbStaires(10)
    fibonacci_4(10)
    end = time.time()
    print('Time cost: {}'.format(end - start))
