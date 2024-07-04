from collections import deque


deque = deque()

print(deque)  # deque([])
deque.append((1, 2))
print(deque)  # deque([(1, 2)])

deque.append((3, 4))
print(deque)  # deque([(1, 2), (3, 4)])

deque.pop()
print(deque)  # deque([(1, 2)])

deque.appendleft((5, 6))
print(deque)  # deque([(5, 6), (1, 2)])
