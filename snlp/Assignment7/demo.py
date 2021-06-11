from importlib import reload
import exercise_1
exercise_1 = reload(exercise_1)

tree = exercise_1.CountTree(n=4)

assert tree.get("") == 0
tree.add("ABCE")
tree.add("ABCD")
tree.add("ABCD")
tree.add("QBCD")
tree.add("QQCD")
tree.add("BCDA")
tree.add("1234")
tree.add("1234")
tree.add("1234")
tree.add("1234")
tree.add("1234")
tree.add("5634")
print(tree.get("ABCD"))
assert tree.get("ABCD") == 2
assert tree.get("ABCX") == 0
assert tree.get("BCD") == 3
assert tree.get("D") == 4
assert tree.get("CD") == 4
assert tree.get("1234") == 5
assert tree.get("5634") == 1
tree.prune(4)
assert tree.get("ABCD") == 4
assert tree.get("XXCD") == 4
assert tree.get("D") == 4
assert tree.get("1234") == 5
assert tree.get("5634") == 1