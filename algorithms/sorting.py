



def bubble_sort(nums: list, order='asc') -> list:
    # stable O(n^2)
    for i in range(len(nums) - 1):
        flag = True
        for j in range(len(nums) - 1):
            if nums[j] > nums[j + 1] and order == 'asc':
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
                flag = False

            elif nums[j] < nums[j + 1] and order == 'decs':
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
                flag = False
        if flag:
            break
    return nums


def selection_sorting(nums: list, order='acs') -> list:
    '''
    1. 首先在未排序序列中找到最小（大）元素，存放到排序序列的起始位置

    2. 再从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾。

    3. 重复第二步，直到所有元素均排序完毕。
    '''

    for i in range(len(nums)):
        min = i
        for j in range(i + 1, len(nums)):
            if nums[j] < nums[min]:
                min = j
                nums[min], nums[i] = nums[i], nums[min]
            # pop the last one, which is the smallest one
    return nums


def insertion_sort(nums, order='acs'):
    '''
    插入排序（Insertion Sort）是一种简单直观的排序算法。它的工作原理是通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。
    从第一个元素开始，该元素可以认为已经被排序
    取出下一个元素，在已经排序的元素序列中从后向前扫描
    如果该元素（已排序）大于新元素，将该元素移到下一位置
    重复步骤3，直到找到已排序的元素小于或者等于新元素的位置
    将新元素插入到该位置后
    重复步骤2~5
    '''
    for i in range(1, len(nums)):
        
        if nums[i] < nums[i - 1]:
            tmp = nums[i]
            index = i

            for j in range(i - 1, -1, -1):
                if tmp < nums[j]:
                    nums[j + 1] = nums[j]
                    index = j
            
            nums[index] = tmp
    return nums


def shell_sorting(nums):
    # unstable
    gap = len(nums) // 2
    while gap > 0:
        for i in range(gap, len(nums)):
            tmp = nums[i]
            j = i
            while j >= gap and nums[j - gap] > tmp:
                nums[j] = nums[j - gap]
                j -= gap
            nums[j] = tmp
        gap = gap // 2
    return nums


def merge_sorting(nums):
    '''
    递归法，分区，每个区最多两个数，然后排序，，然后merge，一直merge
    '''
    pass


def quick_sorting(nums):
    
    pivot_list = list()
    left_list, right_list = list(), list()
    if len(nums) <= 1:
        return nums
    pivot = nums[0]
    for item in nums:
        if item < pivot:
            left_list.append(item)
        elif item > pivot:
            right_list.append(item)
        else:
            pivot_list.append(item)
    

        

    left_list = quick_sorting(left_list)
    right_list = quick_sorting(right_list)

    return left_list + pivot_list + right_list


def quick_sorting_simplified(nums):
    if len(nums) <= 1:
        return nums
    pivot = nums[0]
    left_list = [item for item in nums[1:] if item < pivot]
    right_list = [item for item in nums[1:] if item >= pivot]

    return quick_sorting_simplified(left_list) + [pivot] + quick_sorting_simplified(right_list)


def heap_sorting(nums):
    pass


if __name__ == "__main__":
    nums = [3, 5, 1, 6, 2, 4, 5]
    # nums = [6,5,4,3,2,1]
    print(quick_sorting_simplified(nums))
