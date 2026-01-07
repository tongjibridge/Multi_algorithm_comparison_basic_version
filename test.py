# 读取test.xslx，以第一列为横坐标，第二列为纵坐标
class a:
    def __init__(self, x, y, z, v):
        self.z = z
        self.v = v
        self.x = x
        self.y = y

    def sum(self):
        return self.x + self.y + self.z + self.v


paras1 = {
    "x": 1,
    "y": 2,
}
paras2 = {
    "z": 3,
    "v": 4,
}
obj = a(**paras1, **paras2)
print(obj.sum())
