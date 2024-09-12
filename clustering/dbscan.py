# DBSCAN

import datahelper


# 判断是不是已经分好簇了
def in_groups(_x, _y):
    for xys in groups.values():
        for p_x, p_y in xys:

            if p_x == _x and p_y == _y:
                return True
    return False


# 获取一个点所有的邻居
def get_neighbors(_x, _y):
    _neighbors = []
    for p_x, p_y in data:
        if p_x == _x and p_y == _y:  # 自己不是自己的邻居
            continue
        if (p_x - _x) ** 2 + (p_y - _y) ** 2 <= distance_thresh:
            _neighbors.append([p_x, p_y])

    return _neighbors


def get_grouped_pts():
    _count = 0
    for xys in groups.values():
        _count += len(xys)
    return _count


def get_ungrouped_p():
    for p_x, p_y in data:
        if not in_groups(p_x, p_y):
            return p_x, p_y
    return None


# groups
min_pts = 13
distance_thresh = 2.35
groups = {}
# 存放聚类信息，格式（x,y）:[[x1,y1],[x2,y2]...],xy是这个簇中最先被选中的点
data = datahelper.generatorN(5)  # 生成原始数据
judged = []  # 判断某个点有没有被判断过
count = 0
while len(data) != get_grouped_pts():
    x, y = get_ungrouped_p()
    stack = [[x, y]]
    groups[(x, y)] = [[x, y]]
    # 把这个点可以连接到的点加入整个簇
    while len(stack) != 0:
        x_, y_ = stack.pop()
        neighbors = get_neighbors(x_, y_)
        if len(neighbors) >= min_pts:
            for neighbor in neighbors:
                if neighbor not in judged:
                    stack.append(neighbor)
                    judged.append(neighbor)
            for neighbor in neighbors:
                if not in_groups(*neighbor):
                    groups[(x, y)].append(neighbor)
datahelper.draw_data(groups)