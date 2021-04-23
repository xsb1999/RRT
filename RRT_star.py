import matplotlib.pyplot as plt
import random
import math
import copy
import time
import numpy as np
from shapely.geometry import Point, LineString

show_animation = True


class Node(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        # parent是index，不是node！！！
        self.parent = None


class RRT_star(object):

    def __init__(self, start, goal, obstacle_list, rand_area, search_radius):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:random sampling Area [min,max]
        """
        self.start = Node(start[0], start[1])
        self.end = Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expandDis = 0.8
        self.goalSampleRate = 0.01  # 选择终点的概率是0.01
        self.maxIter = 1000
        self.obstacleList = obstacle_list
        self.nodeList = [self.start]
        self.search_radius = search_radius

    # 产生随机节点
    def random_node(self):
        node_x = random.uniform(self.min_rand, self.max_rand)
        node_y = random.uniform(self.min_rand, self.max_rand)
        node = [node_x, node_y]

        return node

    # 返回距离随机点最近的点在node List中的index
    def get_nearest_list_index(self, node_list, rnd):
        d_list = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 for node in node_list]
        min_index = d_list.index(min(d_list))
        return min_index

    # 寻找new_node附近半径为search_radius的所有nodes
    def find_near_neighbor(self, new_node):
        index_list = []
        for i in range(len(self.nodeList)):
            if self.get_distance(new_node, self.nodeList[i]) <= self.search_radius:  # 还没考虑碰撞！
                index_list.append(i)
        return index_list

    # 检测是否有collision
    def collision_check(self, new_node, obstacle_list):
        a = 0
        for (ox, oy, size) in obstacle_list:
            dx = ox - new_node.x
            dy = oy - new_node.y
            d = math.sqrt(dx * dx + dy * dy)
            if d <= size:
                a = 1  # 有collision

        return a

    # 检测是否在剪枝时有collision(两点连线穿过障碍了)
    def collision_check_prune(self, point1, point2):
        line = LineString([point1, point2])
        for (ox, oy, size) in self.obstacleList:
            if line.distance(Point(ox, oy)) <= size:
                return False  # 两点连线穿过障碍了
        return True

    def new_state(self, rnd, nearest_node, nodeList):
        # 返回弧度制
        theta = math.atan2(rnd[1] - nearest_node.y, rnd[0] - nearest_node.x)
        new_node = copy.deepcopy(nearest_node)
        new_node.x += self.expandDis * math.cos(theta)  # expandDis 是长度，cos是方向
        new_node.y += self.expandDis * math.sin(theta)
        new_node.parent = nodeList.index(nearest_node)
        return new_node

    def planning(self, length):
        i = 0
        while i <= self.maxIter:
            # Random Sampling
            if random.random() > self.goalSampleRate:
                rnd = self.random_node()
            else:
                rnd = [self.end.x, self.end.y]

            # Find nearest node
            min_index = self.get_nearest_list_index(self.nodeList, rnd)
            nearest_node = self.nodeList[min_index]
            # get new node
            new_node = self.new_state(rnd, nearest_node, self.nodeList)

            if self.collision_check(new_node, self.obstacleList):
                print("collision")
                continue

            print('right')

            # star
            neighbor_index = self.find_near_neighbor(new_node)
            self.nodeList.append(new_node)
            if len(neighbor_index) > 0:
                self.choose_parent(new_node, neighbor_index)
                self.rewire(new_node, neighbor_index)

            # check goal
            dx = new_node.x - self.end.x
            dy = new_node.y - self.end.y
            d = math.sqrt(dx * dx + dy * dy)
            if d <= 0.5:
                print("Done!")
                break
            # 画图
            # 可以显示每步（树的生成过程）但要注意内存，障碍多了运行时间长了pycharm就卡了，图片就不动了（卡死了）
            # 如果卡死了就把值改大一点（下面的24，即间隔更多步之后显示一次图）即可
            if i % 24 == 0:  # 可以将24改大一些以防图片卡死 (如果想看得更细一点，即每步一动，就把这个数值调小一些即可)
                self.draw_graph(rnd, new_node)
            i = i + 1
        path = [[self.end.x, self.end.y]]
        last_index = len(self.nodeList) - 1
        # while循环从终点向前追溯路线
        while self.nodeList[last_index].parent is not None:
            node = self.nodeList[last_index]
            path.append([node.x, node.y])
            last_index = node.parent
        path.append([self.start.x, self.start.y])

        return path, i

    # 按照cost选取cost最小的父节点
    def choose_parent(self, new_node, neighbor_index):
        cost = [self.get_new_cost(self.nodeList[i], new_node) for i in neighbor_index]
        cost_min_index = neighbor_index[int(np.argmin(cost))]
        new_node.parent = cost_min_index

    def rewire(self, new_node, neighbor_index):
        for i in neighbor_index:
            node_neighbor = self.nodeList[i]
            # 比较是否需要更新父节点
            if self.cost(node_neighbor) > self.get_new_cost(new_node, node_neighbor):
                node_neighbor.parent = self.nodeList.index(new_node)

    # 两部分（父节点到原点+new_node到父节点）
    def get_new_cost(self, node_start, node_end):
        dist = self.get_distance(node_start, node_end)
        return self.cost(node_start) + dist

    # 两点间的欧氏距离
    def get_distance(self, node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy)

    # node到原点距离
    def cost(self, node_p):
        node = node_p
        cost = 0.0
        while node.parent:
            cost += math.hypot(node.x - self.nodeList[node.parent].x, node.y - self.nodeList[node.parent].y)
            node = self.nodeList[node.parent]
        return cost

    # 画图
    def draw_graph(self, rnd=None, new_node=None):
        plt.clf()
        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^g")
        for node in self.nodeList:
            if node.parent is not None:
                plt.plot([node.x, self.nodeList[node.parent].x], [
                    node.y, self.nodeList[node.parent].y], "-g")

        for (ox, oy, size) in self.obstacleList:
            plt.plot(ox, oy, "sk", ms=10 * size)

        plt.plot(self.start.x, self.start.y, "pr", ms=15)
        plt.plot(self.end.x, self.end.y, "pb", ms=15)
        x_circle = new_node.x
        y_circle = new_node.y
        r = self.search_radius
        circle = plt.Circle((x_circle, y_circle), r, color='r', fill=False)
        plt.gcf().gca().add_artist(circle)
        plt.axis('equal')

        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
        plt.grid(True)
        plt.pause(0.5)

    def draw_static(self, path, path_prune):
        plt.clf()
        for node in self.nodeList:
            if node.parent is not None:
                plt.plot([node.x, self.nodeList[node.parent].x], [
                    node.y, self.nodeList[node.parent].y], "-g")

        for (ox, oy, size) in self.obstacleList:
            plt.plot(ox, oy, "sk", ms=10 * size)

        plt.plot(self.start.x, self.start.y, "pr", ms=15)
        plt.plot(self.end.x, self.end.y, "pb", ms=15)
        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])

        plt.plot([data[0] for data in path], [data[1] for data in path], '*-r')
        plt.plot([data[0] for data in path_prune], [data[1] for data in path_prune], '*-m', linewidth=3)
        plt.grid(True)
        plt.show()

    # 剪枝
    def Pruning(self, path):
        unidirectionalPath = [path[0]]
        pointTem = path[0]
        for i in range(2, len(path)):
            point1 = (pointTem[0], pointTem[1])
            point2 = (path[i][0], path[i][1])
            if not self.collision_check_prune(point1, point2):
                pointTem = path[i - 1]
                unidirectionalPath.append(pointTem)
        unidirectionalPath.append(path[-1])
        return unidirectionalPath


def main():
    print("start RRT path planning...")
    # 创建障碍物
    obstacle_list = [
        (5, 1, 3),
        (3, 6, 2),
        (3, 8, 2),
        (1, 1, 2),
        (3, 5, 2),
        (9, 5, 2),
        (8, 1, 5),
        (5, 5, 2),
        (-1, 5, 1),
        (7, 7, 1)]

    plt.clf()
    length = len(obstacle_list)
    rrt = RRT_star(start=[0, 0], goal=[8, 9], rand_area=[-2, 11], obstacle_list=obstacle_list, search_radius=1)
    start = time.time()
    path, i = rrt.planning(length)
    end = time.time()
    print('#############################')
    print('total trying times: ' + str(i))
    print('total running time: ' + str(end - start) + 's')
    print('normal path: ' + str(path))
    # 剪枝
    path_prune = rrt.Pruning(path)
    print('pruning path: ' + str(path_prune))

    # 画出最终路线
    if show_animation:
        plt.close()
        rrt.draw_static(path, path_prune)


if __name__ == '__main__':
    main()
