# -*- coding: utf-8 -*-
"""八叉树构建与区域查询（仅使用 Python 标准库）。

公共接口（名字与签名保持骨架约定）：

- :class:`AABB`、:class:`Node`
- :func:`build_octree`
- :func:`query_sphere`、:func:`query_box`
- :func:`load_points`

另附：:class:`QueryResult`（``list`` 子类，携带候选节点数等统计）、
:func:`tree_to_dict` / :func:`tree_from_dict`（序列化往返）、
:func:`count_nodes` / :func:`tree_depth`（结构统计）、
:func:`brute_force_sphere` / :func:`brute_force_box`（暴力对照）。
"""

import json
from dataclasses import dataclass, field


@dataclass
class AABB:
    """轴对齐包围盒，区间为左闭右闭 ``[lo, hi]``。"""

    lo: tuple
    hi: tuple


@dataclass
class Node:
    """八叉树节点。

    - 叶子节点：``children == []``，点保存在 ``points`` 中；
    - 内部节点：``children`` 为长度 8 的列表（按卦限索引，空卦限为 ``None``），
      此时 ``points`` 必须为空——分裂后父节点不再持有叶子数据。
    """

    box: AABB
    depth: int = 0
    children: list = field(default_factory=list)
    points: list = field(default_factory=list)


class QueryResult(list):
    """查询命中点列表（``list`` 子类，可直接当列表用）。

    额外携带遍历统计，用于证明查询走的是索引而非全表扫描：

    - ``candidate_nodes``：AABB 与查询区域相交、被实际访问的节点数（候选节点数）；
    - ``points_tested``：在候选叶子中真正做几何判定的点数；
    - ``nodes_visited``：遍历过程中访问过的全部节点数（含不相交被剪枝节点）。
    """

    def __init__(self, hits=(), candidate_nodes=0, points_tested=0,
                 nodes_visited=0):
        super().__init__(hits)
        self.candidate_nodes = candidate_nodes
        self.points_tested = points_tested
        self.nodes_visited = nodes_visited

    def __repr__(self):
        return ("QueryResult(hits=%d, candidate_nodes=%d, points_tested=%d, "
                "nodes_visited=%d)" % (len(self), self.candidate_nodes,
                                       self.points_tested, self.nodes_visited))


# --------------------------------------------------------------------------- #
# 建树
# --------------------------------------------------------------------------- #

def _mid(lo, hi):
    """盒中心。整数边界返回 float，保证分裂后仍是浮点比较。"""
    return (lo + hi) / 2.0


def _octant_index(point, center):
    """点相对盒中心所在的卦限索引（x/y/z 各占一位，0 取负侧、1 取正侧）。"""
    idx = 0
    for k, value in enumerate(point):
        if value >= center[k]:
            idx |= 1 << k
    return idx


def _child_box(box, idx):
    """卦限 ``idx`` 对应的子盒；中心面由相邻两个子盒共享（边界点有确定归属）。"""
    center = tuple(_mid(lo, hi) for lo, hi in zip(box.lo, box.hi))
    lo = tuple(center[k] if (idx >> k) & 1 else box.lo[k]
               for k in range(len(center)))
    hi = tuple(box.hi[k] if (idx >> k) & 1 else center[k]
               for k in range(len(center)))
    return AABB(lo, hi)


def _root_box(points):
    """由点集确定根盒；空点集返回以原点为中心的单位盒（保证空树结构确定）。"""
    if not points:
        return AABB((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    dim = len(points[0])
    lo = [min(p[k] for p in points) for k in range(dim)]
    hi = [max(p[k] for p in points) for k in range(dim)]
    # 某条轴完全重合时，给出一个非零宽度，避免子盒退化成同一个盒。
    for k in range(dim):
        if lo[k] == hi[k]:
            lo[k] -= 1.0
            hi[k] += 1.0
    return AABB(tuple(lo), tuple(hi))


def build_octree(points, capacity, max_depth):
    """构建八叉树。

    参数
    ----
    points: 可迭代的三维点（元组/列表均可），内部统一转为元组；
    capacity: 叶子容量，超过容量且未到 ``max_depth`` 时必须分裂；
    max_depth: 最大深度（根为 0）；到达该深度后叶子不再分裂。

    同一点集、同一参数反复构建得到结构完全相同的树：点的归属只由坐标与
    盒边界决定，与输入顺序无关；叶子内的点按坐标排序，因此连序列化文本
    也是字节级一致的。
    """
    if not isinstance(capacity, int) or capacity < 0:
        raise ValueError("capacity 必须为非负整数")
    if not isinstance(max_depth, int) or max_depth < 0:
        raise ValueError("max_depth 必须为非负整数")

    pts = sorted(tuple(p) for p in points)  # 排序：结构与输入顺序无关
    root = Node(box=_root_box(pts), depth=0)

    def build(node, node_points):
        node.points = list(node_points)
        if len(node_points) <= capacity or node.depth >= max_depth:
            return  # 叶子：容量内或已达最大深度
        # 超容量必须分裂
        buckets = [[] for _ in range(8)]
        center = tuple(_mid(lo, hi)
                       for lo, hi in zip(node.box.lo, node.box.hi))
        for p in node_points:
            buckets[_octant_index(p, center)].append(p)
        node.children = [None] * 8
        node.points = []  # 分裂后父节点不再持有叶子数据
        for idx, bucket in enumerate(buckets):
            if bucket:
                child = Node(box=_child_box(node.box, idx),
                             depth=node.depth + 1)
                node.children[idx] = child
                build(child, bucket)

    build(root, pts)
    return root


# --------------------------------------------------------------------------- #
# 几何判定
# --------------------------------------------------------------------------- #

def _aabb_intersect(a, b):
    """两个闭区间 AABB 是否相交。"""
    return all(a.lo[k] <= b.hi[k] and b.lo[k] <= a.hi[k]
               for k in range(len(a.lo)))


def _aabb_distance_sq(box, center):
    """点 ``center`` 到 AABB 的最近距离平方（点在盒内为 0）。"""
    total = 0.0
    for k, c in enumerate(center):
        if c < box.lo[k]:
            delta = box.lo[k] - c
        elif c > box.hi[k]:
            delta = c - box.hi[k]
        else:
            delta = 0.0
        total += delta * delta
    return total


def _point_in_sphere(point, center, r):
    """点是否落在闭球内（含球面，r=0 时只命中与中心重合的点）。"""
    return sum((a - b) ** 2 for a, b in zip(point, center)) <= r * r


def _point_in_box(point, box):
    """点是否落在闭盒内（含盒面/盒边/盒角）。"""
    return all(box.lo[k] <= point[k] <= box.hi[k]
               for k in range(len(point)))


# --------------------------------------------------------------------------- #
# 查询
# --------------------------------------------------------------------------- #

def query_sphere(root, center, r):
    """半径（闭球）查询，返回 :class:`QueryResult`。

    仅下钻 AABB 与球相交的节点；每个点只属于一个叶子，因此不会重复命中。
    """
    if r < 0:
        raise ValueError("半径 r 不能为负")
    center = tuple(center)
    hits = []
    stats = {"candidate_nodes": 0, "points_tested": 0, "nodes_visited": 0}

    def visit(node):
        stats["nodes_visited"] += 1
        # AABB 与球不相交则整支剪枝
        if _aabb_distance_sq(node.box, center) > r * r:
            return
        stats["candidate_nodes"] += 1
        if not node.children:  # 叶子
            for p in node.points:
                stats["points_tested"] += 1
                if _point_in_sphere(p, center, r):
                    hits.append(p)
            return
        for child in node.children:  # 固定卦限顺序，结果确定
            if child is not None:
                visit(child)

    visit(root)
    return QueryResult(hits, stats["candidate_nodes"],
                       stats["points_tested"], stats["nodes_visited"])


def query_box(root, box):
    """盒子（闭 AABB）查询，返回 :class:`QueryResult`。"""
    if not isinstance(box, AABB):
        box = AABB(tuple(box.lo), tuple(box.hi))
    hits = []
    stats = {"candidate_nodes": 0, "points_tested": 0, "nodes_visited": 0}

    def visit(node):
        stats["nodes_visited"] += 1
        if not _aabb_intersect(node.box, box):  # 整支剪枝
            return
        stats["candidate_nodes"] += 1
        if not node.children:  # 叶子
            for p in node.points:
                stats["points_tested"] += 1
                if _point_in_box(p, box):
                    hits.append(p)
            return
        for child in node.children:
            if child is not None:
                visit(child)

    visit(root)
    return QueryResult(hits, stats["candidate_nodes"],
                       stats["points_tested"], stats["nodes_visited"])


# --------------------------------------------------------------------------- #
# 结构统计与序列化
# --------------------------------------------------------------------------- #

def count_nodes(root):
    """树中节点总数（含根）。"""
    total = 1

    def walk(node):
        nonlocal total
        for child in node.children:
            if child is not None:
                total += 1
                walk(child)

    walk(root)
    return total


def tree_depth(root):
    """树的最大深度（根深度为 0；空树只有根，返回 0）。"""
    deepest = 0

    def walk(node):
        nonlocal deepest
        if node.depth > deepest:
            deepest = node.depth
        for child in node.children:
            if child is not None:
                walk(child)

    walk(root)
    return deepest


def tree_to_dict(root):
    """把树序列化为纯 JSON 友好的嵌套 dict。"""

    def convert(node):
        data = {
            "box": {"lo": list(node.box.lo), "hi": list(node.box.hi)},
            "depth": node.depth,
            "points": [list(p) for p in node.points],
            # 内部节点固定 8 槽（空卦限为 None），叶子为空列表
            "children": ([None] * 8 if node.children else []),
        }
        if node.children:
            data["children"] = [
                convert(child) if child is not None else None
                for child in node.children
            ]
        return data

    return convert(root)


def tree_from_dict(data):
    """由 :func:`tree_to_dict` 的输出重建树。"""

    def convert(d):
        node = Node(
            box=AABB(tuple(d["box"]["lo"]), tuple(d["box"]["hi"])),
            depth=d["depth"],
            points=[tuple(p) for p in d["points"]],
        )
        if d["children"]:
            node.children = [
                convert(child) if child is not None else None
                for child in d["children"]
            ]
        return node

    return convert(data)


def dumps_tree(root):
    """序列化为 JSON 字符串（键序固定，两次输出字节级一致）。"""
    return json.dumps(tree_to_dict(root), sort_keys=True,
                      ensure_ascii=False, separators=(",", ":"))


def loads_tree(text):
    """读取 :func:`dumps_tree` 产生的 JSON 字符串。"""
    return tree_from_dict(json.loads(text))


# --------------------------------------------------------------------------- #
# IO 与暴力对照
# --------------------------------------------------------------------------- #

def load_points(path):
    """从 JSON 文件读入点集，返回元组列表。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [tuple(p) for p in data]


def brute_force_sphere(points, center, r):
    """逐点暴力半径查询（含球面），作为正确性基准。"""
    center = tuple(center)
    return [tuple(p) for p in points
            if _point_in_sphere(tuple(p), center, r)]


def brute_force_box(points, box):
    """逐点暴力盒子查询（含边界），作为正确性基准。"""
    if not isinstance(box, AABB):
        box = AABB(tuple(box.lo), tuple(box.hi))
    return [tuple(p) for p in points if _point_in_box(tuple(p), box)]


def insert_point(root, point):
    """插入一个点，返回新的根或原地更新；分裂规则与构建一致。"""
    raise NotImplementedError("insert_point 尚未实现")


def remove_point(root, point):
    """删除一个点；空节点回收，树深不膨胀。"""
    raise NotImplementedError("remove_point 尚未实现")


def tree_stats(root):
    """返回 dict(nodes=节点数, depth=树深, leaves=叶子数)。"""
    raise NotImplementedError("tree_stats 尚未实现")


def query_many(root, queries):
    """批量查询，返回每个查询的结果与访问节点数。"""
    raise NotImplementedError("query_many 尚未实现")
