
from math import log
import time
import asyncio
import math

class IterClass(object):
    def __iter__(self):
        self.a = 0
        return self
    
    def __next__(self):
        x = self.a
        self.a += 1
        return x


def consumer():
    r = ''
    while True:
        n = yield r
        if not n:
            return
        print('[CONSUMER] Consuming %s...' % n)
        time.sleep(1)
        r = '200 OK'


def produce(c):
    """
    usage: 
        c = consumer()
        produce(c)
    Args:
        c (generator):
    """
    next(c)
    n = 0
    while n < 5:
        n = n + 1
        print('[PRODUCER] Producing %s...' % n)
        r = c.send(n)
        print('[PRODUCER] Consumer return: %s' % r)


def gen():
    """
    usage:
        g = gen()
        print(next(g))
        print(g.send("world"))

    Yields:
        string:
    """
    s = yield "hello"
    print("用户传递进来的值为：%s" % s)
    yield s


def func_yield_from():
    yield from [1,2,3,4,5]


@asyncio.coroutine
def get_html(url, name):
    print("%s get %s html start" % (name, url))
    yield from asyncio.sleep(2)
    print("%s get %s html end" % (name, url))


def isPalindrome(x: int) -> bool:
    if x < 0:
        return False

    str_x = str(x)
    half_1 = str_x[0: int(len(str_x)/2)]
    half_2 = str_x[-1: -1 - len(half_1): -1]
    return half_1 == half_2


def romanToInt(s='MCMXCIV') -> int:
    all_roman = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    result = 0
    for index in range(len(s)):
        if index < len(s) - 1 and all_roman[s[index]] < all_roman[s[index + 1]]:
            result -= all_roman[s[index]]
        else:
            result += all_roman[s[index]]
    return result


def getExtractedNumber():
    '''
    1. 已知一个A数组包含1～N，例如[1,2,3,4,5,...,100],
    B数组为从A中去除2个元素后并随机打乱顺序的长度为N-2的数组，快速求出这两个数字分别是什么。假设两个数字分别是x， y

    Sum(A) - Sum(B) = x + y
    然后就是两数之和的问题。
    '''
    A = list(range(10))
    # given the 2 extracted numbers is 4 , 5
    # shuffle what left
    B = [1,2,3,6,7,8,9]
    
    # say the 2 numbers are: x, y
    # so x + y = sum(A) - sum(B)
    sum_xy = sum(A) - sum(B)
    tmp = dict()
    for i in range(len(A)):
        x = sum_xy - A[i]
        if tmp.get(x):
            return [tmp.get(x), i]
        tmp[A[i]] = i

def two_sum(nums, target):
    tmp = dict()

    for j in range(len(nums)):
        x = target - nums[j]
        print(tmp.get(x))
        if x in tmp:
            return [tmp.get(x), j]
        tmp[nums[j]] = j

def cup_quality(cup_number, floor_number):
    pass

def superEggDrop(K: int, N: int) -> int:
    # convert to a math problem.
    # (K*(K+1))/2>= N
    memo = {}
    def dynamic_program(k, n):
        if (k, n) not in memo:
            if n == 0:
                ans = 0
            elif k == 1:
                ans = n
            else:
                lo, hi = 1, n
                while lo + 1 < hi:
                    # binary
                    x = (lo + hi) // 2
                    t1 = dynamic_program(k - 1, x - 1)
                    t2 = dynamic_program(k, n - x)

                    if t1 < t2:
                        lo = x
                    elif t1 > t2:
                        hi = x
                    else:
                        lo = hi = x
                ans = 1 + min(
                    max(
                        dynamic_program(k - 1, x - 1), dynamic_program(k, n - x)) for x in (lo, hi)
                        
                    )
            memo[k, n] = ans
        return memo[k, n]
    return dynamic_program(K, N)

def quick_sorting_demo(nums):
    if len(nums) <= 1:
        return nums
    pivot = nums[0]
    left_list = []
    right_list = []
    for i in range(1, len(nums)):
        if nums[i] <= pivot:
            left_list.append(nums[i])
        else:
            right_list.append(nums[i])
    return quick_sorting_demo(left_list) + [pivot] + quick_sorting_demo(right_list)

def coin_change(coins: list, amount: int):
    cache = {}
    def get_coin_changes(amount):
        if amount in cache: return cache[amount]
        if amount == 0: return 0
        
        if amount < 0: return -1
        res = float('INF')
        for value in coins:
            current_count = get_coin_changes(amount - value)
            if current_count == -1: continue
            res = min(res, 1 + current_count)

        cache[amount] = res if res != float('INF') else -1
        return cache[amount]
        
    return get_coin_changes(amount)

def coin_change_2(coins, amount):
    dp_table = [float('INF')] * (amount + 1)
    dp_table[0] = 0
    for coin in coins:
        for x in range(coin, amount + 1):
            dp_table[x] = min(dp_table[x], dp_table[x - coin] + 1)

    return dp_table[amount] if dp_table[amount] != float('INF') else -1

def combinationSum(candidates: list, target: int):

    def dfs(begin, path, new_target):
        if new_target < 0: return
        if new_target == 0:
            res.append(path)
            return
        for index in range(begin, len(candidates)):
            new_new_target = new_target - candidates[index]
            if new_new_target < 0:
                break
            dfs(index, path+[candidates[index]], new_new_target)
    res = []
    path = []
    candidates.sort()
    dfs(0, path, target)
    return res

def combinationSum2(candidates: list, target: int):
    def dfs(begin, path, new_target):
        if new_target < 0: return
        if new_target == 0:
            res.append(path)
            return
        if begin < len(candidates) and candidates[begin] == new_target:
            res.append(path + [candidates[begin]])
            return
        for i in range(begin, len(candidates)):
            if i > begin and candidates[i] == candidates[i-1]:
                continue
            new_new_target = new_target - candidates[i]
            if new_new_target < 0:
                break
            dfs(i+1, path + [candidates[i]], new_new_target)

    res = []
    path = []
    candidates.sort()
    dfs(0, path, target)
    return res

def combo_sum_2(candidates: list, target: int):
    if not candidates: return []
    candidates.sort()
    res = []
    def dfs(begin, path, new_target):
        n = candidates[begin]
        if new_target < n: return
        if new_target == n:
            res.append(path + [n])
        if new_target == 0:
            res.append(path)
            return
        gap = 1
        if begin + gap < len(candidates):
            dfs(begin + gap, path + [n], new_target - n)
        while begin + gap < len(candidates) and candidates[begin + gap] == n:
            gap += 1
        if begin + gap >= len(candidates):
            return
        else:
            dfs(begin + gap, path, new_target)
        
    dfs(0, [], target)
    return res

def combinationSum3(k: int, n: int):
    nums = [i for i in range(1, 10)]
    res = []
    def dfs(begin, path, target):
        if begin > len(nums): return
        if target == 0 and len(path) == k:
            res.append(path)
            return
        for i in range(begin, len(nums)):
            # if i == 4:
            #     print(nums[i], target)
            new_target = target - nums[i]
            if new_target < 0 or len(path) == k:
                break
            dfs(i+1, path + [nums[i]], new_target)
    dfs(0, [], n)
    return res

def combinationSum3_optimized(k: int, n: int):

    res = []
    def dfs(num, path, target):
        if target == 0 and len(path) == k:
            res.append(path)
            return
        if num == 10: return
        if target < 0: return
        if len(path) > k: return
        dfs(num + 1, path + [num], target - num)
        dfs(num + 1, path, target)

    dfs(1, [], n)
    return res

def exist(board: list, word: str) -> bool:
    used = set()

    def recursion(word):
        start_m = 0
        
def permuteUnique(nums):
    res = []
    def dfs(left_nums: list, path):
        if len(left_nums) == 0:
            res.append(path)
            return
        visited = []
        for i in range(len(left_nums)):
            if left_nums[i] in visited:
                continue
            visited.append(left_nums[i])
            dfs(left_nums[0:i] + left_nums[i+1:], path + [left_nums[i]])
    dfs(nums, [])
    return res

def is_number(x):
    try:
        int(x)
        return True
    except Exception as e:
        return False

def calPoints(ops) -> int:
    res = [0,0,0]
    for item in ops:
        if is_number(item):
            res.append(int(item))

        elif item == '+':
            res.append(res[-1] + res[-2])

        elif item == 'C':
            res.pop()
            res.insert(0, 0)
        elif item == 'D':
            res.append(2 * (res[-1]))

    return sum(res)

import collections
def numPairsDivisibleBy60(time) -> int:
    res = 0
    cache = collections.defaultdict(int)
    for t in time:
        if t % 60 in cache:
            res += cache[t % 60]
        if t % 60 == 0:
            res[0] += 1
            continue
        cache[60 - t % 60] += 1
    return res

def moveZeroes(nums) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    index = 0
    count = 0
    while count < len(nums):
        if nums[index] == 0:
            nums.append(nums.pop(index))
        else:
            index += 1
        count += 1

def commonChars(A):
    # a,b,c,d,e,...,z
    res = []
    min_len_word = min(A, key=len)
    for ch in min_len_word:
        if all(ch in item for item in A):
            res.append(ch)
            A = [i.replace(ch,'',1)  for i in A]
    return res

def searchRange(nums, target: int):
    if not nums:
        return [-1, -1]

    if nums[0] > target or nums[-1] < target:
        return [-1, -1]
    left = 0

    nums_len = len(nums)
    right = nums_len - 1
    while True:
        if nums[left] < target:
            left += 1
        if nums[right] > target:
            right -= 1
        if nums[left] == nums[right] and nums[left] == target:
            return [left, right]
        else:
            if right <= left:
                return [-1, -1]

def searchRange2(nums, target: int):
    if not nums or nums[0] > target or nums[-1] < target:
        return [-1, -1]
    left = 0
    right = len(nums)
    
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            start = mid - 1
            end = mid + 1
            while left >= 0 and nums[left] == target:
                start -= 1
            while right < len(nums) and nums[right] == target:
                end += 1
            return [start + 1, end - 1]
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    return [-1, -1]

def lengthOfLongestSubstring(s: str) -> int:
    if not s: return 0
    if len(set(s)) == 1: return 1
    left, right = 0 , 2
    isSubstr = lambda string: len(string) == len(set(string))
    max_length = 1
    while left < len(s) - 1 and right < len(s) + 1:
        if isSubstr(s[left: right]):
            if len(s[left:right]) > max_length:
                max_length = len(s[left:right])
            right += 1
        else:
            left += 1
    return max_length

def groupAnagrams(strs):
    res = list()
    tmp_list = []
    j = 1
    while len(strs) > 0:
        if len(strs) == 1 or j == len(strs):
            tmp_list.append(strs.pop(0))
            res.append(tmp_list)
            tmp_list = []
            continue
        if len(strs[0]) == len(strs[j]) and set(strs[0]) == set(strs[j]):
            # pop to the tmp list
            item = strs.pop(j)
            tmp_list.append(item)
        else:
            if j == len(strs) - 1:
                tmp_list.append(strs.pop(0))
                res.append(tmp_list)
                tmp_list = []
                j = 1
            else:
                j += 1
    return res

def permute(nums):
    # 全排列
    res = []
    def dfs(path, left_nums):
        """
        docstring
        """
        if len(left_nums) == 0:
            res.append(path)
            return
        for i, n in enumerate(left_nums):
            new_left_nums = left_nums[:i] + left_nums[i+1:]
            dfs(path + [n], new_left_nums)
    dfs([], nums)
    return res    
    
def largeGroupPositions(s: str):
    if len(s) < 3:
        return []

    n, num = len(s), 1
    res = []
    for i in range(len(s)):
        if i == n - 1 or s[i] != s[i +1]:
            if num >= 3:
                res.append([i - num + 1, i])
            num = 1
        else:
            num += 1

    return res

def canJump(nums) -> bool:
    index = 0
    def dfs(index, item):
        if index == len(nums) - 1:
            return True
        elif item == 0:
            return False
        for i in range(1, item + 1):
            dfs(index + i, nums[index])

    return dfs(index, nums[0])

def dailyTemperatures(T):
    # binary
    n = len(T)
    res, nxt, big = [0] * n, {}, 10**9
    for i in range(n - 1, -1, -1):
        warmer_index = min(nxt.get(t, big) for t in range(T[i] + 1, 100))
        if warmer_index != big:
            res[i] = warmer_index - i
        nxt[T[i]] = i
    return res

def dailyTemperatures2(T):
    # 单调栈
    stack = []
    res = [0] * len(T)
    for i in range(len(T)):

        while stack and T[stack[-1]] < T[i]:
            top = stack.pop()
            res[top] = i - top

        stack.append(i)

    return res

def isPowerOfTwo(n: int) -> bool:
    bin_n = bin(n)
    if bin_n[:3] == '0b1' and bin_n[3:] == len(bin_n[3:]) * '0':
        return True
    else:
        return False
    
def maxProfit(prices) -> int:
    max_profit = 0
    for i in range(len(prices)):
        # buy
        profits = [p - prices[i] for p in prices[i+1:]]
        profits.append(max_profit)
        max_profit = max(profits)
    return max_profit

def maxCoins(nums) -> int:
    # DFS
    ans = []
    cache = {}
    def dfs(left_nums, res, path):
        if len(left_nums) == 0:
            ans.append(res)
            cache[tuple(path)] = res
            return
        left_nums.append(1)
        left_nums.insert(0, 1)
        for i in range(1, len(left_nums) - 1):
            current_coin = left_nums[i - 1] * left_nums[i] * left_nums[i + 1]
            new_left_nums = left_nums[1: i] + left_nums[i+1: len(left_nums) - 1]
            dfs(new_left_nums, current_coin + res, path + [left_nums[i]])
    dfs(nums, 0, [])
    return max(ans)

def maxIncreaseKeepingSkyline(grid) -> int:
    dim = len(grid)
    left_skyline = [max(grid[i]) for i in range(dim)]
    top_skyline = [max(grid[row][col] for row in range(dim)) for col in range(dim)]
    ans = 0
    for m in range(dim):
        for n in range(dim):
            min_height = min(left_skyline[m], top_skyline[n])
            if grid[m][n] < min_height:
                ans += min_height - grid[m][n]
    return ans

def smallerNumbersThanCurrent(nums):
    sorted_nums = list(sorted(nums))
    print(sorted_nums)
    cache = {}
    for i, val in enumerate(sorted_nums):
        if val in cache:
            continue
        cache[val] = i
    ans = []
    for n in nums:
        ans.append(cache[n])
    return ans

def findLengthOfLCIS(nums) -> int:
    max_len = 1
    right = 1
    base_len = 1
    while right < len(nums):
        if nums[right - 1] < nums[right]:
            base_len += 1
        else:
            base_len = 1
        max_len = max(max_len, base_len)
        right += 1

    return max_len

def sumOddLengthSubarrays(arr) -> int:
    n = len(arr)
    sum_all = 0
    for i in range(1, len(arr) + 1, 2):
        for j in range(n):
            if j + i - 1 >= n:
                break
            sum_all += sum(arr[j: j + i])
    return sum_all

def productExceptSelf(nums):
    n = len(nums)
    ans = [1] * n
    
    for i in range(1, n):
        ans[i] = nums[i - 1] * ans[i - 1]
    print(ans)
    R = 1
    for j in reversed(range(n)):
        ans[j] = ans[j] * R
        R *= nums[j]

    return ans

def subsets(nums):
    ans = []
    def dfs(path, depth, start):
        if len(path) == depth:
            ans.append(path)
            return
        for j, item in enumerate(nums[start:]):
            index = start + j + 1
            dfs(path + [item], depth, index)
    depth = len(nums)
    for i in range(depth + 1):
        dfs([], i, 0)

    return ans

def generateParenthesis(n: int):
    ans = []

    def back_track(path, left, right):
        if len(path) == 2 * n:
            ans.append(''.join(path))
            return
        
        if left < n:
            path.append('(')
            back_track(path, left + 1, right)
            path.pop()
        if right < left:
            path.append(')')
            back_track(path, left, right + 1)
            path.pop()

    
    back_track([], 0, 0)
    return ans

def rotate(matrix) -> None:
    """
    Do not return anything, modify matrix in-place instead.
    """
    n = len(matrix)
    # flip horizontally
    for i in range(n // 2):
        for j in range(n):
            matrix[i][j], matrix[n - 1 - i][j] = matrix[n - 1 - i][j], matrix[i][j]
    # flip by the reversed diagonal
    for i in range(n):
        for j in range(i):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

def totalMoney(n: int) -> int:
    total = 0
    plus = 0
    for i in range(1, n + 1):
        
        total += (i // 7 + i % 8)
        

    return total

def uniquePaths(m: int, n: int) -> int:
    # 动态规划
    dp = [ [0] * n for _ in range(m)]
    # if row == 0
    for i in range(n):
        dp[0][i] = 1
    # if col == 0
    for j in range(m):
        dp[j][0] = 1

    for row in range(1, m):
        for col in range(1, n):
            dp[row][col] = dp[row][col - 1] + dp[row - 1][col]
    return dp[-1][-1]

def findNumOfValidWords(words, puzzles):
    ans = [0] * len(puzzles)
    for i, p in enumerate(puzzles):
        p_set = set(p)
        
        for word in words:
            new_word_set = set(word)
            if word[0] == p[0] and new_word_set.issubset(p_set):
                ans[i] += 1
    return ans

if __name__ == "__main__":

    start = time.time()
    '''
    cases = [
        "abbxxxxzzy",
        "abc",
        "abcdddeeeeaabbbcd",
        "aba",
        "aaa"
    ]
    for case in cases:
        res = largeGroupPositions(case)
        print(res)

    '''
    s = "aaa"
    l1 = ["apple","pleas","please"]
    l2 = ["aelwxyz","aelpxyz","aelpsxy","saelpxy","xaelpsy"]

    print(findNumOfValidWords(l1, l2))
    end = time.time()
    time_cost = end - start
    # average_time_cost = time_cost / len(cases)
    # print('AVERAGE TIME COST: {}'.format(end - start))
    print('TOTAL TIME COST: {}'.format(end - start))
