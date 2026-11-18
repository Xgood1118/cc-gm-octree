# -*- coding: utf-8 -*-
"""八叉树构建、区域查询与动态维护（仅使用 Python 标准库）。

公共接口（名字与签名与骨架保持一致）：

- :class:`AABB` / :class:`Node`
- :func:`build_octree(points, capacity, max_depth)`
- :func:`query_sphere(root, center, r)` / :func:`query_box(root, box)`
- :func:`load_points(path)`
- :func:`insert_point(root, point)` / :func:`remove_point(root, point)`
- :func:`tree_stats(root)` / :func:`build_limited(points, capacity, max_depth, max_nodes)`
- :func:`query_many(root, queries)`

另有序列化、暴力对照与结构校验等辅助接口。

设计约定
--------

* 盒子区间为左闭右闭 ``[lo, hi]``；点落在分裂中心面上时归入坐标取
  "正侧" 的卦限（``x >= center``）。因此每个点唯一属于一个叶子。
* 分裂只发生在 ``点的数量 > capacity`` 且 ``depth < max_depth`` 且该节点
  的点确实能分到至少两个卦限时。内部节点不保存点，点只在叶子中。
* 同一批点、同一组参数构建出的树完全确定：与输入顺序无关，叶子内的点
  按字典序排序，子节点按卦限序号排序。
* 删除采用增量方式：从叶子删点后沿路径向上做 "需要则折叠" 的规范化，
  空子节点立即回收；若剩余点集的根包围盒发生变化（例如删掉了极值点），
  则退化为按剩余点全量重建，从而保证删除结果与全量重建在规范化序列化下
  完全一致。
"""

import json
from collections import deque
from dataclasses import dataclass, field

__all__ = [
    "AABB",
    "Node",
    "QueryResult",
    "build_octree",
    "build_limited",
    "query_sphere",
    "query_box",
    "query_many",
    "query_sphere_stats",
    "query_box_stats",
    "brute_force_sphere",
    "brute_force_box",
    "load_points",
    "insert_point",
    "remove_point",
    "tree_stats",
    "count_nodes",
    "tree_depth",
    "serialize_tree",
    "deserialize_tree",
    "tree_to_dict",
    "tree_from_dict",
    "tree_to_json",
    "tree_from_json",
    "canonical_tree",
    "save_tree",
    "load_tree",
    "validate_tree",
]


@dataclass
class AABB:
    """轴对齐包围盒，区间为左闭右闭 ``[lo, hi]``。"""

    lo: tuple
    hi: tuple

    def contains(self, point):
        """点是否在盒内（边界算命中）。"""
        return all(lo <= v <= hi for lo, hi, v in zip(self.lo, self.hi, point))


@dataclass
class Node:
    """八叉树节点。

    - 叶子节点：``children == []``，点保存在 ``points`` 中（按字典序排序）；
    - 内部节点：``children`` 非空且按卦限序号升序排列，此时 ``points == []``。
    """

    box: AABB
    depth: int = 0
    children: list = field(default_factory=list)
    points: list = field(default_factory=list)


class QueryResult(list):
    """查询命中点列表（``list`` 子类，可直接当列表使用）。

    额外携带遍历统计，用来证明查询走的是索引而非全表扫描：

    - ``candidate_nodes``：包围盒与查询区域相交、被实际遍历的节点数；
    - ``points_tested``：真正做了几何判定的点数；
    - ``nodes_visited``：被检查过包围盒的节点数（含被剪枝的节点）。
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

    def as_dict(self):
        return {
            "hits": [tuple(p) for p in self],
            "candidate_nodes": self.candidate_nodes,
            "points_tested": self.points_tested,
            "nodes_visited": self.nodes_visited,
        }


# --------------------------------------------------------------------------- #
# 参数与几何辅助
# --------------------------------------------------------------------------- #

_CONFIG_ATTR = "_octree_config"
_DEFAULT_CONFIG = {"capacity": 8, "max_depth": 8, "max_nodes": None}


def _validate_params(capacity, max_depth, max_nodes):
    capacity = int(capacity)
    max_depth = int(max_depth)
    if capacity < 1:
        raise ValueError("capacity 必须 >= 1，收到 %r" % (capacity,))
    if max_depth < 0:
        raise ValueError("max_depth 必须 >= 0，收到 %r" % (max_depth,))
    if max_nodes is not None:
        max_nodes = int(max_nodes)
        if max_nodes < 1:
            raise ValueError("max_nodes 必须 >= 1（根节点占 1），收到 %r"
                             % (max_nodes,))
    return capacity, max_depth, max_nodes


def _attach_config(root, capacity, max_depth, max_nodes):
    root.__dict__[_CONFIG_ATTR] = {
        "capacity": capacity,
        "max_depth": max_depth,
        "max_nodes": max_nodes,
    }


def _get_config(root):
    cfg = getattr(root, _CONFIG_ATTR, None)
    if not cfg:
        cfg = _DEFAULT_CONFIG
    return cfg["capacity"], cfg["max_depth"], cfg["max_nodes"]


def _validate_dims(points):
    if not points:
        return 3
    dim = len(points[0])
    if dim < 1:
        raise ValueError("点必须至少有一维")
    for p in points:
        if len(p) != dim:
            raise ValueError("所有点的维度必须一致：期望 %d，收到 %d"
                             % (dim, len(p)))
    return dim


def _midpoint(lo, hi):
    return (lo + hi) / 2.0


def _center(box):
    return tuple(_midpoint(lo, hi) for lo, hi in zip(box.lo, box.hi))


def _octant(point, center):
    """点所在卦限索引：第 k 轴坐标 >= 中心则该位为 1（边界归正侧）。"""
    idx = 0
    for k, value in enumerate(point):
        if value >= center[k]:
            idx |= 1 << k
    return idx


def _child_box(box, idx):
    """卦限 ``idx`` 对应的子盒；中心面被相邻两个子盒共享。"""
    lo, hi = [], []
    for k, (low, high) in enumerate(zip(box.lo, box.hi)):
        mid = _midpoint(low, high)
        if (idx >> k) & 1:
            lo.append(mid)
            hi.append(high)
        else:
            lo.append(low)
            hi.append(mid)
    return AABB(tuple(lo), tuple(hi))


def _octant_of_box(child, parent):
    """由子盒相对父盒中心的位置反推卦限序号。"""
    center = _center(parent.box)
    idx = 0
    for k in range(len(center)):
        if child.box.lo[k] >= center[k]:
            idx |= 1 << k
    return idx


def _root_box(points):
    """由点集确定根盒；空点集返回以原点为中心的单位盒。

    某一维上所有点坐标相同时，向两侧各扩 1，保证子盒有正体积。
    """
    if not points:
        return AABB((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    dim = _validate_dims(points)
    lo = [float(min(p[k] for p in points)) for k in range(dim)]
    hi = [float(max(p[k] for p in points)) for k in range(dim)]
    for k in range(dim):
        if lo[k] == hi[k]:
            lo[k] -= 1.0
            hi[k] += 1.0
    return AABB(tuple(lo), tuple(hi))


def _partition(points, box):
    """按卦限把点分组；返回 ``[(octant_index, sorted_points), ...]``。"""
    center = _center(box)
    buckets = {}
    for p in points:
        buckets.setdefault(_octant(p, center), []).append(p)
    return [(idx, sorted(buckets[idx])) for idx in sorted(buckets)]


def _box_contains(box, point):
    return all(lo <= v <= hi for lo, hi, v in zip(box.lo, box.hi, point))


def _collect_points(node, out=None):
    out = [] if out is None else out
    if node.children:
        for child in node.children:
            _collect_points(child, out)
    else:
        out.extend(node.points)
    return out


# --------------------------------------------------------------------------- #
# 构建
# --------------------------------------------------------------------------- #

def _split_pass(root, capacity, max_depth, max_nodes):
    """广度优先执行分裂；返回建成的节点总数。

    预算 ``max_nodes`` 不足时跳过该次分裂（节点保留为叶子），因此节点数
    永远不超过上限，而查询仍会扫描这些叶子里的全部点，结果保持精确。
    """
    queue = deque([root])
    node_count = 1
    while queue:
        node = queue.popleft()
        if not node.points:
            continue
        if len(node.points) <= capacity or node.depth >= max_depth:
            continue
        parts = _partition(node.points, node.box)
        if len(parts) < 2:
            # 所有点落在同一卦限，继续分裂没有空间收益，保留为叶子。
            continue
        if max_nodes is not None and node_count + len(parts) > max_nodes:
            continue
        node.points = []
        for idx, group in parts:
            child = Node(_child_box(node.box, idx), node.depth + 1, [], group)
            node.children.append(child)
            node_count += 1
        queue.extend(node.children)
    return node_count


def _new_tree(points, capacity, max_depth, max_nodes):
    capacity, max_depth, max_nodes = _validate_params(
        capacity, max_depth, max_nodes)
    pts = [tuple(p) for p in points]
    _validate_dims(pts)
    root = Node(_root_box(pts), 0, [], sorted(pts))
    _attach_config(root, capacity, max_depth, max_nodes)
    _split_pass(root, capacity, max_depth, max_nodes)
    return root


def build_octree(points, capacity, max_depth):
    """构建八叉树。

    :param points: 可迭代的三维点（元组/列表均可）。
    :param capacity: 叶子容量；超出且未到 ``max_depth`` 时必须分裂。
    :param max_depth: 最大深度，根为深度 0。
    """
    return _new_tree(points, capacity, max_depth, None)


def build_limited(points, capacity, max_depth, max_nodes):
    """带节点上限的构建：节点总数不超过 ``max_nodes``。

    达到上限后停止分裂，超容量的叶子保留全部点；查询仍会逐点判定，
    所以结果与暴力枚举一致。
    """
    return _new_tree(points, capacity, max_depth, max_nodes)


# --------------------------------------------------------------------------- #
# 查询
# --------------------------------------------------------------------------- #

def _aabb_intersects(a, b):
    for alo, ahi, blo, bhi in zip(a.lo, a.hi, b.lo, b.hi):
        if alo > bhi or blo > ahi:
            return False
    return True


def _sphere_intersects_box(box, center, radius_sq):
    dist_sq = 0.0
    for lo, hi, value in zip(box.lo, box.hi, center):
        if value < lo:
            delta = lo - value
            dist_sq += delta * delta
        elif value > hi:
            delta = value - hi
            dist_sq += delta * delta
    return dist_sq <= radius_sq


def _box_intersects_box(node_box, query_box):
    return _aabb_intersects(node_box, query_box)


def _traverse(root, box_test, point_test, empty):
    """通用遍历：返回 (hits, candidate_nodes, points_tested, nodes_visited)。"""
    hits = []
    if empty:
        return hits, 0, 0, 1
    if not box_test(root.box):
        return hits, 0, 0, 1
    candidates = 0
    tested = 0
    visited = 1  # 根节点的包围盒已被检查
    stack = [root]
    while stack:
        node = stack.pop()
        candidates += 1
        if node.children:
            for child in reversed(node.children):
                visited += 1
                if box_test(child.box):
                    stack.append(child)
        else:
            for point in node.points:
                tested += 1
                if point_test(point):
                    hits.append(point)
    hits.sort()
    return hits, candidates, tested, visited


def query_sphere(root, center, r):
    """半径查询：返回 ``QueryResult``（list 子类），携带遍历统计。

    距离小于等于 ``r`` 的点都算命中（含边界）；``r < 0`` 时无命中。
    """
    if r < 0:
        return QueryResult()
    center = tuple(center)
    radius_sq = r * r
    empty = not root.points and not root.children

    def box_test(box):
        return _sphere_intersects_box(box, center, radius_sq)

    def point_test(point):
        d = 0.0
        for a, b in zip(point, center):
            diff = a - b
            d += diff * diff
        return d <= radius_sq

    hits, candidates, tested, visited = _traverse(
        root, box_test, point_test, empty)
    return QueryResult(hits, candidates, tested, visited)


def _coerce_box(box):
    if isinstance(box, AABB):
        return box
    lo, hi = box
    return AABB(tuple(lo), tuple(hi))


def query_box(root, box):
    """盒子查询：返回 ``QueryResult``（list 子类），携带遍历统计。

    区间为左闭右闭；``lo > hi`` 的退化盒无命中。
    """
    box = _coerce_box(box)
    empty = not root.points and not root.children
    if any(lo > hi for lo, hi in zip(box.lo, box.hi)):
        return QueryResult()

    def box_test(node_box):
        return _box_intersects_box(node_box, box)

    def point_test(point):
        return all(lo <= v <= hi for lo, hi, v in zip(box.lo, box.hi, point))

    hits, candidates, tested, visited = _traverse(
        root, box_test, point_test, empty)
    return QueryResult(hits, candidates, tested, visited)


def query_sphere_stats(root, center, r):
    """半径查询并以 dict 形式返回结果与统计。"""
    result = query_sphere(root, center, r)
    payload = result.as_dict()
    payload["hits"] = [tuple(p) for p in result]
    return payload


def query_box_stats(root, box):
    """盒子查询并以 dict 形式返回结果与统计。"""
    result = query_box(root, box)
    payload = result.as_dict()
    payload["hits"] = [tuple(p) for p in result]
    return payload


def query_many(root, queries):
    """批量球查询。

    返回与 ``queries`` 等长的 list，每项为::

        {"hits": [...], "visited": 本次访问节点数, "total": 全树节点数,
         "candidate_nodes": ..., "points_tested": ...,
         "saved": 节省的节点数, "saved_ratio": 节省比例,
         "visited_ratio": 访问比例}

    ``visited`` 统计被检查包围盒的节点数（含被剪枝节点），因此
    ``visited <= total``；``saved_ratio == 1 - visited / total``。
    """
    total = count_nodes(root)
    results = []
    for center, radius in queries:
        result = query_sphere(root, center, radius)
        visited = result.nodes_visited
        saved = total - visited
        results.append({
            "hits": [tuple(p) for p in result],
            "visited": visited,
            "total": total,
            "candidate_nodes": result.candidate_nodes,
            "points_tested": result.points_tested,
            "saved": saved,
            "saved_ratio": (saved / total) if total else 0.0,
            "visited_ratio": (visited / total) if total else 0.0,
        })
    return results


def brute_force_sphere(points, center, r):
    """逐点半径查询，用于与索引结果对照。"""
    if r < 0:
        return []
    center = tuple(center)
    radius_sq = r * r
    hits = []
    for p in points:
        p = tuple(p)
        d = 0.0
        for a, b in zip(p, center):
            diff = a - b
            d += diff * diff
        if d <= radius_sq:
            hits.append(p)
    return hits


def brute_force_box(points, box):
    """逐点盒子查询，用于与索引结果对照。"""
    box = _coerce_box(box)
    if any(lo > hi for lo, hi in zip(box.lo, box.hi)):
        return []
    return [tuple(p) for p in points
            if all(lo <= v <= hi for lo, hi, v in zip(box.lo, box.hi, p))]


# --------------------------------------------------------------------------- #
# 动态维护
# --------------------------------------------------------------------------- #

def _find_path(root, point):
    """沿点所在的卦限下降到叶子，返回路径；点不在任何子盒中时返回 None。"""
    path = [root]
    node = root
    while node.children:
        center = _center(node.box)
        idx = _octant(point, center)
        match = None
        for child in node.children:
            if _octant_of_box(child, node) == idx:
                match = child
                break
        if match is None:
            return None
        node = match
        path.append(node)
    return path


def _insert_child_sorted(node, child, idx):
    pos = 0
    while pos < len(node.children):
        if _octant_of_box(node.children[pos], node) > idx:
            break
        pos += 1
    node.children.insert(pos, child)


def _should_collapse(node, points, capacity, max_depth):
    """删除后节点是否应折叠成叶子（与构建规则一致）。"""
    if not points:
        return True
    if len(points) <= capacity or node.depth >= max_depth:
        return True
    return len(_partition(points, node.box)) < 2


def insert_point(root, point):
    """插入一个点，返回根节点；分裂规则与构建一致。

    点落在根盒内时增量下探并只在必要时分裂；点越出根盒、改变了根盒
    形状或树带节点预算时，按全部点重建，保证结构与全量构建一致。
    """
    point = tuple(point)
    capacity, max_depth, max_nodes = _get_config(root)
    empty = not root.children and not root.points
    if max_nodes is not None or empty or not _box_contains(root.box, point):
        pts = _collect_points(root)
        pts.append(point)
        return _new_tree(pts, capacity, max_depth, max_nodes)

    node = root
    while node.children:
        center = _center(node.box)
        idx = _octant(point, center)
        match = None
        for child in node.children:
            if _octant_of_box(child, node) == idx:
                match = child
                break
        if match is None:
            child = Node(_child_box(node.box, idx), node.depth + 1, [], [point])
            _insert_child_sorted(node, child, idx)
            break
        node = match
    else:
        node.points.append(point)
        node.points.sort()
        if len(node.points) > capacity and node.depth < max_depth:
            _split_pass(node, capacity, max_depth, None)

    # 根盒必须等于全部点集的规范包围盒，否则按构建规则重建。
    all_points = _collect_points(root)
    if _root_box(all_points) != root.box:
        return _new_tree(all_points, capacity, max_depth, max_nodes)
    return root


def remove_point(root, point):
    """删除一个点（重复点时只删一个），返回根节点。

    空子节点立即回收；沿路径自底向上折叠不再满足分裂条件的节点，
    因此树深不会因删除而增加。删除后的树与 "用剩余点全量重建" 在
    规范化序列化下完全一致。
    """
    point = tuple(point)
    capacity, max_depth, max_nodes = _get_config(root)
    if max_nodes is not None:
        pts = _collect_points(root)
        try:
            pts.remove(point)
        except ValueError:
            return root
        return _new_tree(pts, capacity, max_depth, max_nodes)

    path = _find_path(root, point)
    if path is None:
        return root
    leaf = path[-1]
    try:
        leaf.points.remove(point)
    except ValueError:
        return root

    # 自底向上：回收空子节点，必要时折叠。
    for node in reversed(path):
        node.children = [c for c in node.children
                         if c.children or c.points]
        if not node.children:
            node.points.sort()
            continue
        points = _collect_points(node)
        if _should_collapse(node, points, capacity, max_depth):
            node.children = []
            node.points = sorted(points)

    remaining = _collect_points(root)
    if not remaining:
        return _new_tree([], capacity, max_depth, None)
    # 根盒随剩余点集的极值变化时，用剩余点重建以保持规范结构。
    if _root_box(remaining) != root.box:
        return _new_tree(remaining, capacity, max_depth, None)
    return root


# --------------------------------------------------------------------------- #
# 统计与校验
# --------------------------------------------------------------------------- #

def tree_stats(root):
    """返回 ``dict(nodes=节点数, depth=树深(层数), leaves=叶子数, points=点数)``。"""
    nodes = 0
    leaves = 0
    points = 0
    max_depth = 0
    stack = [root]
    while stack:
        node = stack.pop()
        nodes += 1
        max_depth = max(max_depth, node.depth)
        if node.children:
            stack.extend(node.children)
        else:
            leaves += 1
            points += len(node.points)
    return {
        "nodes": nodes,
        "depth": max_depth + 1,  # 层数：只有根时为 1
        "leaves": leaves,
        "points": points,
    }


def count_nodes(root):
    return tree_stats(root)["nodes"]


def tree_depth(root):
    return tree_stats(root)["depth"]


def validate_tree(root):
    """校验树的全部结构不变量；不满足时抛 ``ValueError``。"""
    capacity, max_depth, max_nodes = _get_config(root)
    if root.depth != 0:
        raise ValueError("根节点深度必须为 0")

    nodes = 0

    def walk(node, parent, expected_octant):
        nonlocal nodes
        nodes += 1
        if len(node.box.lo) != len(node.box.hi):
            raise ValueError("包围盒维度不一致")
        if any(lo > hi for lo, hi in zip(node.box.lo, node.box.hi)):
            raise ValueError("包围盒 lo > hi")
        for p in node.points:
            if not _box_contains(node.box, p):
                raise ValueError("点 %r 不在节点包围盒内" % (p,))
        if node.points != sorted(node.points):
            raise ValueError("叶子内的点未排序")
        if parent is not None:
            if node.depth != parent.depth + 1:
                raise ValueError("子节点深度错误")
            if _octant_of_box(node, parent) != expected_octant:
                raise ValueError("子节点卦限与包围盒不一致")
        if node.children:
            if node.points:
                raise ValueError("内部节点不得持有叶子数据")
            if len(node.children) < 2:
                raise ValueError("内部节点至少要有两个非空子节点")
            seen = set()
            order = []
            for child in node.children:
                idx = _octant_of_box(child, node)
                if idx in seen:
                    raise ValueError("同一卦限出现多个子节点")
                seen.add(idx)
                order.append(idx)
                if child.box != _child_box(node.box, idx):
                    raise ValueError("子节点包围盒与父节点不匹配")
                if not child.children and not child.points:
                    raise ValueError("存在空叶子（未回收）")
                walk(child, node, idx)
            if order != sorted(order):
                raise ValueError("子节点未按卦限升序排列")
        else:
            if parent is not None and not node.points:
                raise ValueError("存在空叶子（未回收）")
            if len(node.points) > capacity and node.depth < max_depth:
                if max_nodes is None:
                    if len(_partition(node.points, node.box)) >= 2:
                        raise ValueError("超容量叶子未分裂")
                # 有预算时允许因预算耗尽而保留超容量叶子。

    walk(root, None, None)
    if max_nodes is not None and nodes > max_nodes:
        raise ValueError("节点数 %d 超过上限 %d" % (nodes, max_nodes))
    return True


# --------------------------------------------------------------------------- #
# 序列化
# --------------------------------------------------------------------------- #

def _node_to_dict(node):
    return {
        "box": {"lo": [float(v) for v in node.box.lo],
                "hi": [float(v) for v in node.box.hi]},
        "depth": node.depth,
        "points": [list(p) for p in node.points],
        "children": [_node_to_dict(c) for c in node.children],
    }


def _node_from_dict(data):
    box = AABB(tuple(data["box"]["lo"]), tuple(data["box"]["hi"]))
    node = Node(box, int(data.get("depth", 0)), [],
                [tuple(p) for p in data.get("points", [])])
    node.children = [_node_from_dict(c) for c in data.get("children", [])]
    return node


def serialize_tree(root):
    """把树序列化为可 JSON 化的 dict（含构建参数）。"""
    capacity, max_depth, max_nodes = _get_config(root)
    return {
        "version": 1,
        "config": {
            "capacity": capacity,
            "max_depth": max_depth,
            "max_nodes": max_nodes,
        },
        "root": _node_to_dict(root),
    }


def deserialize_tree(data):
    """由 :func:`serialize_tree` 的输出还原树。"""
    root = _node_from_dict(data["root"])
    cfg = data.get("config") or _DEFAULT_CONFIG
    _attach_config(root,
                   cfg.get("capacity", _DEFAULT_CONFIG["capacity"]),
                   cfg.get("max_depth", _DEFAULT_CONFIG["max_depth"]),
                   cfg.get("max_nodes"))
    return root


def tree_to_dict(root):
    """``serialize_tree`` 的别名。"""
    return serialize_tree(root)


def tree_from_dict(data):
    """``deserialize_tree`` 的别名。"""
    return deserialize_tree(data)


def tree_to_json(root, **kwargs):
    kwargs.setdefault("sort_keys", True)
    kwargs.setdefault("ensure_ascii", False)
    return json.dumps(serialize_tree(root), **kwargs)


def tree_from_json(text):
    return deserialize_tree(json.loads(text))


def save_tree(path, root):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialize_tree(root), f, ensure_ascii=False,
                  sort_keys=True, indent=2)
    return path


def load_tree(path):
    with open(path, "r", encoding="utf-8") as f:
        return deserialize_tree(json.load(f))


def _canonical_node_dict(node):
    """规范化节点表示：点与子节点都显式排序，便于跨路径比较。"""
    children = sorted(node.children, key=lambda c: (tuple(c.box.lo),
                                                    tuple(c.box.hi)))
    return {
        "box": {"lo": [float(v) for v in node.box.lo],
                "hi": [float(v) for v in node.box.hi]},
        "depth": node.depth,
        "points": [list(p) for p in sorted(tuple(p) for p in node.points)],
        "children": [_canonical_node_dict(c) for c in children],
    }


def canonical_tree(root):
    """规范化序列化（JSON 字符串）。

    与输入顺序无关；用于比较两次构建、序列化往返以及删除与全量重建
    的结果是否完全一致。
    """
    capacity, max_depth, max_nodes = _get_config(root)
    payload = {
        "version": 1,
        "config": {
            "capacity": capacity,
            "max_depth": max_depth,
            "max_nodes": max_nodes,
        },
        "root": _canonical_node_dict(root),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# 读入
# --------------------------------------------------------------------------- #

def load_points(path):
    """从 JSON 文件读入点集。

    支持顶层为点列表，或 ``{"points": [...]}`` 形式。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("points", [])
    return [tuple(p) for p in data]
