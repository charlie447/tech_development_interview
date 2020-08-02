
class Stack(object):
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if self.is_empty():
            return "Error: empty stack."
        return self.items.pop()

    def is_empty(self):
        return self.size() == 0

    def size(self):
        return len(self.items)

    def peek(self):
        return self.items[-1]


# 栈的应用

# 1. 符号匹配
# 单种符号匹配.左括号时入栈，右括号时出栈
# example_char_string = ['((((((())','()))','(()()(()','()()()','(()()']
def check_character_balance(char_string):
    stack = Stack()

    for i in range(len(char_string)):
        if char_string[i] == '(':
            stack.push(char_string[i])
        else:
            if not  stack.is_empty():
                stack.pop()
            else:
                return False
        
    return stack.is_empty()


# 2. 十进制转换为二进制

# 3. 算术表达式转换

# Fibonacci sequence 斐波那契数列
def fibonacci(n):
    """generate Fibonacci sequence according to following function
    F(n) = 0 when n = 0
    F(n) = 1 when n = 1
    F(n) = F(n-1) + F(n-2) when n > 1

    Args:
        n (inte): the length of the Fibonacci sequence that about to generate
    """

    if n == 0:
        return 0
    
    if n == 1:
        return 1

    if n > 1:
        return fibonacci(n - 1) + fibonacci(n - 2)

    

if __name__ == "__main__":
    example_char_string = ['((((((())','()))','(()()(()','()()()','(()()']
    
    # print(fibonacci(6))
    for n in range(1, 10):
        print(fibonacci(n))
