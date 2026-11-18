# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import octree as mod  # noqa: E402

octree = mod
AABB = mod.AABB


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #

def grid_points(n, step=1.0, origin=0.0):
    """n x n x n 网格点。"""
    return [(origin + x * step, origin + y * step, origin + z * step)
            for x in range(n) for y in range(n) for z in range(n)]


def all_leaves(root):
    leaves = []

    def walk(node):
        if not node.children:
            leaves.append(node)
        else:
            for child in node.children:
                if child is not None:
                    walk(child)

    walk(root)
    return leaves


def internal_nodes(root):
    out = []

    def walk(node):
        if node.children:
            out.append(node)
            for child in node.children:
                if child is not None:
                    walk(child)

    walk(root)
    return out


# --------------------------------------------------------------------------- #
# 骨架原有测试（保留名字与断言）
# --------------------------------------------------------------------------- #

def test_build_and_query():
    pts = octree.load_points("data/points.json")
    root = octree.build_octree(pts, 4, 6)
    hits = octree.query_sphere(root, (0, 0, 0), 2.5)
    brute = [p for p in pts
             if sum((a - b) ** 2 for a, b in zip(p, (0, 0, 0))) <= 6.25]
    assert len(hits) == len(brute)


def test_empty_tree():
    root = octree.build_octree([], 4, 6)
    assert octree.query_sphere(root, (0, 0, 0), 1.0) == []


# --------------------------------------------------------------------------- #
# 构建：分裂语义
# --------------------------------------------------------------------------- #

def test_split_parents_hold_no_leaf_data():
    """超容量必须分裂，且分裂后父节点不再持有叶子数据。"""
    pts = grid_points(4)
    root = octree.build_octree(pts, capacity=4, max_depth=6)

    for node in internal_nodes(root):
        assert node.points == [], "内部节点不应持有任何点"
        assert len(node.children) == 8

    # 所有点仍可在叶子中找到，且没有丢失
    leaf_points = [p for leaf in all_leaves(root) for p in leaf.points]
    assert sorted(leaf_points) == sorted(pts)

    # 超容量的叶子不应存在（除非触及最大深度）
    for leaf in all_leaves(root):
        if leaf.depth < 6:
            assert len(leaf.points) <= 4


def test_max_depth_prevents_split():
    """到达最大深度后即使超容量也不再分裂。"""
    pts = [(0.0, 0.0, 0.0)] * 10  # 重复点无法靠分裂分开
    root = octree.build_octree(pts, capacity=2, max_depth=3)
    leaves = all_leaves(root)
    assert all(leaf.depth <= 3 for leaf in leaves)
    deepest = [leaf for leaf in leaves if leaf.depth == 3]
    assert deepest and all(len(leaf.points) > 2 for leaf in deepest)
    assert octree.tree_depth(root) == 3


def test_build_is_deterministic():
    """同一批点连跑两次得到完全相同的树结构（含打乱输入顺序）。"""
    pts = grid_points(3, step=0.7, origin=-3.0)
    root_a = octree.build_octree(pts, 4, 6)
    root_b = octree.build_octree(list(reversed(pts)), 4, 6)

    assert octree.dumps_tree(root_a) == octree.dumps_tree(root_b)
    assert octree.count_nodes(root_a) == octree.count_nodes(root_b)
    # 叶子点序也完全一致（构建与输入顺序无关）
    leaves_a = [tuple(leaf.points) for leaf in all_leaves(root_a)]
    leaves_b = [tuple(leaf.points) for leaf in all_leaves(root_b)]
    assert leaves_a == leaves_b


# --------------------------------------------------------------------------- #
# 序列化往返
# --------------------------------------------------------------------------- #

def test_serialization_roundtrip():
    pts = grid_points(3)
    root = octree.build_octree(pts, 4, 6)
    text = octree.dumps_tree(root)

    restored = octree.loads_tree(text)
    assert octree.dumps_tree(restored) == text  # 再序列化字节级一致
    assert octree.count_nodes(restored) == octree.count_nodes(root)
    assert octree.tree_depth(restored) == octree.tree_depth(root)

    # 读回的树查询结果与原树一致
    q1 = octree.query_sphere(root, (1.0, 1.0, 1.0), 1.5)
    q2 = octree.query_sphere(restored, (1.0, 1.0, 1.0), 1.5)
    assert sorted(q1) == sorted(q2)
    assert q1.candidate_nodes == q2.candidate_nodes


def test_serialization_empty_tree():
    root = octree.build_octree([], 4, 6)
    restored = octree.loads_tree(octree.dumps_tree(root))
    assert octree.count_nodes(restored) == 1
    assert restored.points == [] and restored.children == []
    assert octree.query_box(restored, AABB((-1, -1, -1), (1, 1, 1))) == []


# --------------------------------------------------------------------------- #
# 查询与暴力一致
# --------------------------------------------------------------------------- #

def test_sphere_query_matches_brute_force():
    pts = grid_points(5, step=1.0, origin=-2.0)
    root = octree.build_octree(pts, 4, 8)
    center, r = (0.3, -0.7, 1.1), 2.3

    result = octree.query_sphere(root, center, r)
    brute = octree.brute_force_sphere(pts, center, r)
    assert sorted(result) == sorted(brute)


def test_box_query_matches_brute_force():
    pts = grid_points(5, step=1.0, origin=-2.0)
    root = octree.build_octree(pts, 4, 8)
    box = AABB((-1.5, -1.5, 0.0), (2.0, 1.5, 2.0))

    result = octree.query_box(root, box)
    brute = octree.brute_force_box(pts, box)
    assert sorted(result) == sorted(brute)


def test_query_visits_only_candidate_nodes():
    """候选节点/实测点数显著少于全表，证明没有全表扫描且无重复命中。"""
    pts = grid_points(6)  # 216 个点
    root = octree.build_octree(pts, 4, 8)
    assert octree.count_nodes(root) < len(pts)

    result = octree.query_sphere(root, (1.0, 1.0, 1.0), 0.8)
    assert len(result) == len(set(result))  # 无重复命中
    assert result.points_tested < len(pts)  # 没有逐点全表扫描
    assert result.candidate_nodes < octree.count_nodes(root)

    box_result = octree.query_box(
        root, AABB((0.5, 0.5, 0.5), (1.5, 1.5, 1.5)))
    assert box_result.points_tested < len(pts)
    assert box_result.candidate_nodes < octree.count_nodes(root)


# --------------------------------------------------------------------------- #
# 边界点
# --------------------------------------------------------------------------- #

def test_points_on_node_boundaries_have_deterministic_owner():
    """落在分裂面上的点必须全部被收入且归属唯一（取正侧卦限）。"""
    pts = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (-1.0, -1.0, -1.0),
           (0.0, 1.0, -1.0), (1.0, 0.0, 0.0)]
    root = octree.build_octree(pts, 2, 6)

    leaf_points = [p for leaf in all_leaves(root) for p in leaf.points]
    assert sorted(leaf_points) == sorted(pts)
    assert len(leaf_points) == len(pts)  # 每个点恰好出现一次


def test_sphere_boundary_inclusive():
    """球面/盒面上的点按闭区间命中，结果确定。"""
    pts = grid_points(3)  # 坐标 0,1,2
    root = octree.build_octree(pts, 4, 6)

    # 点 (2,0,0) 到中心 (0,0,0) 距离恰好为 2
    result = octree.query_sphere(root, (0.0, 0.0, 0.0), 2.0)
    assert (2.0, 0.0, 0.0) in result
    brute = octree.brute_force_sphere(pts, (0.0, 0.0, 0.0), 2.0)
    assert sorted(result) == sorted(brute)

    # r=0 只命中与中心重合的点（含重复点的每一份）
    zero = octree.query_sphere(root, (1.0, 1.0, 1.0), 0.0)
    assert list(zero) == [(1.0, 1.0, 1.0)]


def test_box_boundary_inclusive():
    pts = grid_points(4)
    root = octree.build_octree(pts, 4, 6)
    box = AABB((1.0, 1.0, 1.0), (2.0, 2.0, 2.0))  # 面上的网格点也要命中

    result = octree.query_box(root, box)
    brute = octree.brute_force_box(pts, box)
    assert sorted(result) == sorted(brute)
    assert (1.0, 2.0, 1.0) in result and (2.0, 2.0, 2.0) in result
    assert (0.999, 0.0, 0.0) not in result


# --------------------------------------------------------------------------- #
# 重复点
# --------------------------------------------------------------------------- #

def test_duplicate_points_kept_and_counted():
    pts = [(0.0, 0.0, 0.0)] * 5 + [(1.0, 0.0, 0.0)] * 3 + [(3.0, 3.0, 3.0)]
    root = octree.build_octree(pts, 2, 6)

    leaf_points = [p for leaf in all_leaves(root) for p in leaf.points]
    assert sorted(leaf_points) == sorted(pts)
    assert len(leaf_points) == len(pts)

    result = octree.query_sphere(root, (0.0, 0.0, 0.0), 0.5)
    assert len(result) == 5  # 5 份重复点全部命中，不去重
    assert result.count((0.0, 0.0, 0.0)) == 5

    box = AABB((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    box_hits = octree.query_box(root, box)
    assert len(box_hits) == 8
    assert len(octree.brute_force_box(pts, box)) == 8
    assert sorted(box_hits) == sorted(octree.brute_force_box(pts, box))


def test_duplicate_points_roundtrip():
    pts = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
    root = octree.build_octree(pts, 1, 6)
    restored = octree.loads_tree(octree.dumps_tree(root))
    hits = octree.query_sphere(restored, (0.0, 0.0, 0.0), 0.1)
    assert len(hits) == 2


# --------------------------------------------------------------------------- #
# 空树与参数校验
# --------------------------------------------------------------------------- #

def test_empty_tree_queries_and_stats():
    root = octree.build_octree([], 4, 6)
    assert octree.count_nodes(root) == 1
    assert octree.tree_depth(root) == 0

    sphere = octree.query_sphere(root, (5.0, 5.0, 5.0), 1.0)
    box = octree.query_box(root, AABB((-10, -10, -10), (10, 10, 10)))
    assert list(sphere) == [] and list(box) == []
    # 根盒与球不相交：访问了根但整支剪枝，候选节点为 0
    assert sphere.candidate_nodes == 0 and sphere.nodes_visited == 1
    assert sphere.points_tested == 0
    # 盒子完全包住根盒：根成为候选叶子并被检查
    assert box.candidate_nodes == 1 and box.points_tested == 0


def test_invalid_arguments():
    import pytest
    with pytest.raises(ValueError):
        octree.build_octree([(0.0, 0.0, 0.0)], -1, 6)
    with pytest.raises(ValueError):
        octree.build_octree([(0.0, 0.0, 0.0)], 4, -1)
    root = octree.build_octree([(0.0, 0.0, 0.0)], 4, 6)
    with pytest.raises(ValueError):
        octree.query_sphere(root, (0.0, 0.0, 0.0), -1.0)


# ===== v2 契约测试：初始全部失败，实现后必须通过 =====
import octree as broadphase_oct  # noqa: E402


def test_insert_matches_rebuild():
    base = [(0.1, 0.2, 0.3), (0.9, 0.8, 0.7), (0.5, 0.5, 0.5)]
    root = broadphase_oct.build_octree(list(base), 2, 6)
    root = broadphase_oct.insert_point(root, (0.3, 0.3, 0.3))
    rebuilt = broadphase_oct.build_octree(list(base) + [(0.3, 0.3, 0.3)], 2, 6)
    assert broadphase_oct.dumps_tree(root) == broadphase_oct.dumps_tree(rebuilt)


def test_remove_matches_rebuild_and_shrinks():
    pts = [(0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.9, 0.9, 0.9), (0.75, 0.15, 0.5)]
    root = broadphase_oct.build_octree(list(pts), 1, 8)
    before = broadphase_oct.tree_stats(root)
    root = broadphase_oct.remove_point(root, (0.9, 0.9, 0.9))
    after = broadphase_oct.tree_stats(root)
    rebuilt = broadphase_oct.build_octree([p for p in pts if p != (0.9, 0.9, 0.9)], 1, 8)
    assert broadphase_oct.dumps_tree(root) == broadphase_oct.dumps_tree(rebuilt)
    assert after["depth"] <= before["depth"]


def test_max_nodes_bounds_growth_keeps_correctness():
    pts = [(i / 20.0, (i * 3 % 20) / 20.0, (i * 7 % 20) / 20.0) for i in range(20)]
    root = broadphase_oct.build_octree(pts, 1, 12, max_nodes=9)
    st = broadphase_oct.tree_stats(root)
    assert st["nodes"] <= 9
    res = broadphase_oct.query_sphere(root, (0.5, 0.5, 0.5), 0.4)
    assert sorted(res) == sorted(broadphase_oct.brute_force_sphere(pts, (0.5, 0.5, 0.5), 0.4))


def test_query_many_matches_single():
    pts = [(i / 10.0, (i * i % 10) / 10.0, 0.5) for i in range(10)]
    root = broadphase_oct.build_octree(pts, 2, 6)
    qs = [((0.2, 0.2, 0.5), 0.2), ((0.8, 0.8, 0.5), 0.3)]
    many = broadphase_oct.query_many(root, qs)
    for q, r in zip(qs, many):
        single = broadphase_oct.query_sphere(root, q[0], q[1])
        assert sorted(r["hits"]) == sorted(single)
        assert r["visited"] <= r["total"]
