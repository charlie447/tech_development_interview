class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class BstTree(TreeNode):
    def __init__(self, data):
        self.bst_insert(data)

    def bst_insert(self, data):
        if self.data is None:
            self.data = data
            return
        if data < self.data:
            if self.left is None:
                self.left = TreeNode(data)
            else:
                # recursive
                self.left.bst_insert(data)
        else:
            if self.right is None:
                self.right = TreeNode(data)
            else:
                self.right.bst_insert(data)

class Solution:
    def increasingBST(self, root: TreeNode) -> TreeNode:
        pass

if __name__ == "__main__":
    solution = Solution()
    tree_data = [5,3,6,2,4,None,8,1,None,None,None,7,9]

    bst = BstTree(tree_data)
    

