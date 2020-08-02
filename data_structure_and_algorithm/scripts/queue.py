
class Queue(object):

    def __init__(self):
        self.items = []

    def enqueue(self, item):

        self.items.append(item)

    def dequeue(self):
        if self.is_empty():
            return "Error: the queue is empty."
        return self.items.pop(0)

    def is_empty(self):
        return self.size() == 0

    def size(self):
        return len(self.items)


def josephus(namelist, num):
    # 在 约瑟夫斯问题 中，参与者围成一个圆圈，从某个人（队首）开始报数，报数到n+1的人退出圆圈，然后从退出人的下一位重新开始报数；重复以上动作，直到只剩下一个人为止。
    # pop the num + 1 name till the queue has only one left
    queue = Queue()
    for name in namelist:
        queue.enqueue(name)

    while queue.size() > 1:
        for i in range(num):
            # 队列首位移动到队列尾部
            queue.enqueue(queue.dequeue())
        queue.dequeue()

    return queue.dequeue()


if __name__ == "__main__":
    name_list = ["Bill", "David", "Kent", "Jane", "Susan", "Brad"]
    num = 3
    print(josephus(name_list, num))
        