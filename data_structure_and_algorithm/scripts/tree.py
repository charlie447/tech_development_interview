import functools
import collections


def is_empty_tree(func):
    @functools.wraps(func)
    def wrapper(self):
        if not self.root or self.root.data is None:
            return []
        else:
            return func(self)
    return wrapper

class Node(object):
    def __init__(self, data=None, key=None, value=None, left_child=None, right_child=None) -> None:
        self.data = data
        self.left_child = left_child
        self.right_child = right_child
        # used in AVL
        self.depth = 0
        self.parent = None
        # bs as Balance Factor
        self.bf = 0
        self.key = key
        self.value = value
        # for leetcode
        self.left = None
        self.right = None
        self.val = data
    
    def bst_insert(self, data):
        if self.data is None:
            self.data = data
            return
        if data < self.data:
            if self.left_child is None:
                self.left_child = Node(data)
            else:
                # recursive
                self.left_child.bst_insert(data)
        else:
            if self.right_child is None:
                self.right_child = Node(data)
            else:
                self.right_child.bst_insert(data)

def depth(child_tree: Node):
    if not child_tree:
        return 0
    return 1 + max(depth(child_tree.left_child), depth(child_tree.right_child))

def node_bf(child_tree: Node):
    if not child_tree or not child_tree.data:
        return 0
    bf = abs(depth(child_tree.left_child) - depth(child_tree.right_child))
    return bf

def is_balance(child_tree: Node):
    # O(N^2)
    bf = node_bf(child_tree)
    if bf > 1:
        return False
    else:
        return is_balance(child_tree.left_child) and is_balance(child_tree.right_child)

class BinaryTree(object):
    def __init__(self, node=None) -> None:
        self.root = node
    
    def add(self, item):
        node = Node(item)
        # if the current tree is a empty tree
        if not self.root or self.root.data is None:
            self.root = node
        else:
            # add nodes by left to right
            node_queue = []
            node_queue.append(self.root)

            while True:
                current_node = node_queue.pop(0)
                if current_node.data is None:
                    # placeholder for root.
                    continue
                if not current_node.left_child:
                    # add a left child
                    current_node.left_child = node
                    return
                elif not current_node.right_child:
                    current_node.right_child = node
                    return
                else:
                    node_queue.append(current_node.left_child)
                    node_queue.append(current_node.right_child)
    
    @is_empty_tree
    def floor_travel(self):
        # BFS Broad first 
        tmp_queue = []
        return_queue = []
        tmp_queue.append(self.root)
        while tmp_queue:
            current_node = tmp_queue.pop(0)
            return_queue.append(current_node)
            if current_node.left_child:
                tmp_queue.append(current_node.left_child)
            if current_node.right_child:
                tmp_queue.append(current_node.right_child)
        return return_queue
    
    @is_empty_tree
    def levelOrderBottom(self):
        # leetcode 107
        if not self.root:
            return []
        level_list = []
        stack = collections.deque([self.root])
        while stack:
            level = list()
            for _ in range(len(stack)):
                node = stack.popleft()
                level.append(node.data)
                if node.left_child:
                    stack.append(node.left_child)
                if node.right_child:
                    stack.append(node.right_child)
            level_list.append(level)
        return level_list[::-1]



    @is_empty_tree
    def front_travel(self):
        '''
        Using stack, which is a better way than using loop
        root -> left -> right
        '''
        stack = []
        queue = []
        stack.append(self.root)
        while stack:
            current_node = stack.pop()
            queue.append(current_node)
            # The differences to BFS
            if current_node.right_child:
                stack.append(current_node.right_child)
            if current_node.left_child:
                stack.append(current_node.left_child)
        return queue
    
    def front_travel_with_loop_1(self, root):
        if root == None:
            return []
        print(root.data, end=' ')
        self.front_travel_with_loop_1(root.left_child)
        self.front_travel_with_loop_1(root.right_child)
    
    @is_empty_tree
    def front_travel_with_loop_2(self):
        queue = []
        def loop(root):
            if not root:
                return
            queue.append(root)
            loop(root.left_child)
            loop(root.right_child)
        loop(self.root)
        return queue


    @is_empty_tree
    def middle_travel(self):
        '''
        left -> root -> right
        '''
        stack = []
        queue = []
        tmp_list = []
        stack.append(self.root)
        while stack:
            current_node = stack.pop()
            if current_node.right_child and current_node.right_child not in stack:
                stack.append(current_node.right_child)
            if current_node.left_child:
                if current_node not in tmp_list:
                    tmp_list.append(current_node)
                    stack.append(current_node)
                else:
                    tmp_list.remove(current_node)
                    queue.append(current_node)
                    continue
                stack.append(current_node.left_child)
            else:
                queue.append(current_node)
        return queue
                
    @is_empty_tree
    def back_travel(self):
        '''
        left -> right -> root
        '''
        stack = []
        queue = []
        tmp_list = []
        stack.append(self.root)
        while stack:
            current_node = stack[-1]
            if current_node.right_child and current_node not in tmp_list:
                stack.append(current_node.right_child)
            if current_node.left_child and current_node not in tmp_list:
                stack.append(current_node.left_child)
            if current_node in tmp_list or (current_node.left_child is None and current_node.right_child is None):
                queue.append(stack.pop())
                if current_node in tmp_list:
                    tmp_list.remove(current_node)
            tmp_list.append(current_node)
        
        return queue

    def depth(self, child_root):
        if not child_root:
            return 0
        
        return 1 + max(self.depth(child_root.left_child), self.depth(child_root.right_child))

class BalancedBinaryTree(object):
    def __init__(self, node=None) -> None:
        self.root = node

    def add(self, **item):
        """插入数据
        1 寻找插入点,并记录下距离该插入点的最小非平衡子树及其父子树
        2 修改最小非平衡子树到插入点的bf
        3 进行调整

        Args:
            item (any): the data of a node
        """
        node = Node(**item)
        if self.root is None or self.root.data is None:
            self.root = node
            return
        # the least unbalanced node
        non_balance_node = self.root
        insert_to_node = self.root
        non_balance_node_parent = None
        insert_node_parent = None
        while insert_to_node:
            if node.data == insert_to_node.data:
                return
            # if the insertion node is non-balanced
            if insert_to_node.bf != 0:
                non_balance_node_parent, non_balance_node = insert_node_parent, insert_to_node
            insert_node_parent = insert_to_node

            if node.data > insert_to_node.data:
                insert_to_node = insert_to_node.right_child
            else:
                insert_to_node = insert_to_node.left_child
            # loop until the insert_to_node is None,
            # which means that the final insertion position has been found.
        if node.data > insert_node_parent.data:
            # insert to the right
            insert_node_parent.right_child = node
        else:
            insert_node_parent.left_child = node
        
        # update bf
        tmp_non_balance = non_balance_node
        while tmp_non_balance:
            if node.data == tmp_non_balance.data:
                break
            if node.data > tmp_non_balance.data:
                tmp_non_balance.bf -= 1
                tmp_non_balance = tmp_non_balance.right_child
            else:
                tmp_non_balance.bf += 1
                tmp_non_balance = tmp_non_balance.left_child
        
        # get what side of non_balance_node that the  the new point inserted to.
        # True repr the left, False repr the right.
        if non_balance_node.data > node.data:
            insert_position = non_balance_node.left_child.data > node.data
        else:
            insert_position = non_balance_node.right_child.data > node.data

        # Rotate to maintain balance
        if non_balance_node.bf > 1:

            if insert_position:
                non_balance_node = BalancedBinaryTree.right_rotate(non_balance_node)
            else:
                non_balance_node = BalancedBinaryTree.right_left_rotate(non_balance_node)
        elif non_balance_node.bf < -1:
            if insert_position:
                non_balance_node = BalancedBinaryTree.left_right_rotate(non_balance_node)
            else:
                non_balance_node = BalancedBinaryTree.left_rotate(non_balance_node)
        # assign the non_balance_node to the parent or root which depends on the node data.
        if non_balance_node_parent:
            if non_balance_node_parent.data > non_balance_node.data:
                non_balance_node_parent.left_child = non_balance_node
            else:
                non_balance_node_parent.right_child = non_balance_node
        else:
            self.root = non_balance_node
            

    @staticmethod
    def left_rotate(node):
        node.bf = node.right_child.bf = 0

        node_right = node.right_child
        node.right_child = node.right_child.left_child
        node_right.left_child = node
        return node_right

    @staticmethod
    def right_rotate(node):
        node.bf = node.left_child.bf = 0

        node_left = node.left_child
        node.left_child = node.left_child.right_child
        node_left.right_child = node
        return node_left
            

    @staticmethod
    def left_right_rotate(node):
        node_b = node.left_child
        node_c = node_b.right_child
        node.left_child = node_c.right_child
        node_b.right_child = node_c.left_child
        node_c.left_child = node_b

        node_c.right_child = node
        # update bf
        if node_c.bf == 0:
            node.bf = node_b.bf = 0
        elif node_c.bf == 1:
            node.bf = -1
            node_b.bf = 0
        else:
            node.bf = 0
            node_b.bf = 1

        node_c.bf = 0
        return node_c

    @staticmethod
    def right_left_rotate(node):
        node_b = node.right_child
        node_c = node_b.left_child

        node_b.left_child = node_c.right_child
        node.right_child = node_c.left_child
        node_c.right_child = node_b

        node_c.left_child = node

        if node_c.bf == 0:
            node.bf = node_b.bf = 0
        elif node_c.bf == 1:
            node.bf = 0
            node_b.bf = -1
        else:
            node.bf = 1
            node_b.bf = 0

        node_c.bf = 0
        return node_c

    

class BinarySortTree(object):
    def __init__(self, node=None) -> None:
        self.root = node
    
    def add(self, item):
        """插入数据

        Args:
            item (any): the data of a node
        """

        node = Node(item)
        if self.root is None or self.root.data is None:
            self.root = node
            return
        
        node_queue = []
        node_queue.append(self.root)
        while node_queue:
            current_node = node_queue.pop()
            if node.data >= current_node.data:
                if current_node.right_child:
                    node_queue.append(current_node.right_child)
                else:
                    current_node.right_child = node
            else:
                if current_node.left_child:
                    node_queue.append(current_node.left_child)
                else:
                    current_node.left_child = node
        

class RedBlackTree(object):
    def __init__(self, node=None) -> None:
        self.root = node

    def add(self, **item):
        pass
    

if __name__ == "__main__":
    #创建一个二叉树对象
    node_list = [3,9,20,None,None,15,7]
    tree = BinaryTree()
    #以此向树中添加节点，i == 3的情况表示，一个空节点，看完后面就明白了
    for i in node_list:
        tree.add(i)
    print(tree.levelOrderBottom())
    #广度优先的层次遍历算法
    # print([i.data for i in tree.floor_travel()])
    #前，中，后序 遍历算法（栈实现）
    # print([i.data for i in tree.front_travel()])
    # tree.front_travel_with_loop_1(tree.root)
    # print([i.data for i in tree.front_travel_with_loop_2()])
    
    # print([i.data for i in tree.middle_travel()])
    # print([i.data for i in tree.back_travel()])
    print('------------------------------------')
    #前，中，后序 遍历算法（堆栈实现）
    # print([i.data for i in tree.front_stank_travel()])
    # print([i.data for i in tree.middle_stank_travel()])
    # print([i.data for i in tree.back_stank_travel()])

