
class Node(object):

    def __init__(self, data) -> None:
        self.data = data
        self.prev = None
        self.next = None


class SingleLinkList(object):

    def __init__(self) -> None:
        self._head = None


class BilateralLinkList(object):

    def __init__(self) -> None:
        self._head = None
        self._tail = None
    
    def is_empty(self):
        pass

    def length(self):
        pass

    def items(self):
        pass

if __name__ == "__main__":
    single_link_list = SingleLinkList()

    node_1 = Node(1)
    node_2 = Node(2)
    
    single_link_list._head = node_1
    node_1.next = node_2

    print(single_link_list._head.data)
    print(single_link_list._head.next.data)

