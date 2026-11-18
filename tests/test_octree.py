# -*- coding: utf-8 -*-
"""八叉树索引器测试：构建、查询、边界、序列化、动态维护、预算与批量查询。"""

import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import octree as mod  # noqa: E402
import octree as octree  # noqa: E402  (与骨架契约测试相同的导入方式)


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #

def _pts(n=12):
    return [((i * 7 % 13) / 13.0, (i * 5 % 11) / 11.0, (i * 3 % 7) / 7.0)
            for i in range(n)]


def _all_points(node):
    if node.children:
        out = []
        for c in node.children:
            out.extend(_all_points(c))
        return out
    return list(node.points)


def _walk(node):
    yield node
    for c in node.children:
        yield from _walk(c)


def _sorted(points):
    return sorted(tuple(p) for p in points)


def _rand_points(rng, n, grid=False):
    if grid:
        choices = (0.0, 0.5, 1.0)
        return [(rng.choice(choices), rng.choice(choices), rng.choice(choices))
                for _ in range(n)]
    return [(rng.random(), rng.random(), rng.random()) for _ in range(n)]


# --------------------------------------------------------------------------- #
# 原有契约
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


def test_insert_keeps_structure_consistent():
    pts = _pts()
    root = octree.build_limited(pts, 2, 8, 64)
    root = octree.insert_point(root, (0.33, 0.44, 0.55))
    st = octree.tree_stats(root)
    assert st["nodes"] >= 1 and st["depth"] >= 1
    res = octree.query_sphere(root, (0.33, 0.44, 0.55), 0.05)
    assert (0.33, 0.44, 0.55) in [tuple(p) for p in res]


def test_remove_shrinks_and_stays_queryable():
    pts = _pts()
    root = octree.build_limited(pts, 1, 10, 200)
    before = octree.tree_stats(root)
    root = octree.remove_point(root, pts[-1])
    after = octree.tree_stats(root)
    assert after["depth"] <= before["depth"]
    res = octree.query_sphere(root, pts[-1], 0.01)
    assert pts[-1] not in [tuple(p) for p in res]


def test_build_limited_bounds_nodes():
    pts = _pts(24)
    root = octree.build_limited(pts, 1, 14, 9)
    assert octree.tree_stats(root)["nodes"] <= 9
    assert _sorted(octree.query_sphere(root, (0.5, 0.5, 0.5), 0.5)) == _sorted(
        octree.brute_force_sphere(pts, (0.5, 0.5, 0.5), 0.5))


def test_query_many_matches_single():
    pts = _pts(10)
    root = octree.build_limited(pts, 2, 8, 64)
    qs = [((0.2, 0.2, 0.2), 0.2), ((0.8, 0.8, 0.8), 0.3)]
    many = octree.query_many(root, qs)
    for q, r in zip(qs, many):
        assert _sorted(r["hits"]) == _sorted(
            octree.query_sphere(root, q[0], q[1]))
        assert r["visited"] <= r["total"]


# --------------------------------------------------------------------------- #
# 1. 构建
# --------------------------------------------------------------------------- #

def test_split_respects_capacity_and_depth():
    pts = _pts(40)
    root = octree.build_octree(pts, 2, 3)
    for node in _walk(root):
        if node.children:
            assert node.points == [], "分裂后父节点不得持有叶子数据"
            assert len(node.children) >= 2
            assert all(c.depth == node.depth + 1 for c in node.children)
        else:
            # 到达深度上限前，叶子容量必须被满足（或所有点挤在同一卦限）
            if node.depth < 3:
                centers = node.box
                same_octant = len({
                    tuple(p[k] >= (centers.lo[k] + centers.hi[k]) / 2
                          for k in range(3)) for p in node.points}) == 1
                assert len(node.points) <= 2 or same_octant


def test_max_depth_zero_keeps_single_leaf():
    root = octree.build_octree(_pts(30), 1, 0)
    assert root.children == []
    assert len(root.points) == 30
    assert octree.tree_stats(root)["nodes"] == 1


def test_capacity_is_configurable():
    pts = _pts(40)
    shallow = octree.build_octree(pts, 32, 8)
    deep = octree.build_octree(pts, 1, 8)
    assert octree.tree_stats(deep)["nodes"] > octree.tree_stats(shallow)["nodes"]


def test_build_is_deterministic_and_order_independent():
    pts = _pts(50) + _pts(20)
    a = octree.build_octree(pts, 3, 5)
    b = octree.build_octree(pts, 3, 5)
    c = octree.build_octree(list(reversed(pts)), 3, 5)
    assert octree.canonical_tree(a) == octree.canonical_tree(b)
    assert octree.canonical_tree(a) == octree.canonical_tree(c)
    assert json.dumps(octree.serialize_tree(a), sort_keys=True) == \
        json.dumps(octree.serialize_tree(b), sort_keys=True)


def test_no_point_is_lost_or_duplicated_by_split():
    pts = _pts(60) + [(0.5, 0.5, 0.5)] * 3
    root = octree.build_octree(pts, 1, 10)
    assert _sorted(_all_points(root)) == _sorted(pts)


def test_serialization_roundtrip_text_and_structure():
    root = octree.build_octree(_pts(35), 2, 6)
    text = octree.tree_to_json(root)
    restored = octree.tree_from_json(text)
    assert octree.canonical_tree(restored) == octree.canonical_tree(root)
    assert text == octree.tree_to_json(restored)
    octree.validate_tree(restored)
    # 读回后查询结果与原树一致
    center, r = (0.4, 0.4, 0.4), 0.3
    assert _sorted(octree.query_sphere(restored, center, r)) == \
        _sorted(octree.query_sphere(root, center, r))


def test_serialization_roundtrip_file(tmp_path):
    root = octree.build_octree(_pts(20), 3, 5)
    path = tmp_path / "tree.json"
    octree.save_tree(str(path), root)
    restored = octree.load_tree(str(path))
    assert octree.canonical_tree(restored) == octree.canonical_tree(root)
    octree.validate_tree(restored)


def test_serialization_carries_config_for_dynamic_ops():
    root = octree.build_octree(_pts(20), 2, 5)
    restored = octree.tree_from_json(octree.tree_to_json(root))
    inserted = octree.insert_point(restored, (0.11, 0.22, 0.33))
    full = octree.build_octree(_pts(20) + [(0.11, 0.22, 0.33)], 2, 5)
    assert octree.canonical_tree(inserted) == octree.canonical_tree(full)


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        octree.build_octree(_pts(3), 0, 3)
    with pytest.raises(ValueError):
        octree.build_octree(_pts(3), 2, -1)
    with pytest.raises(ValueError):
        octree.build_limited(_pts(3), 2, 3, 0)


# --------------------------------------------------------------------------- #
# 2. 查询与暴力一致
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", range(8))
def test_sphere_matches_brute_force(seed):
    rng = random.Random(seed)
    pts = _rand_points(rng, 120, grid=(seed % 2 == 0))
    root = octree.build_octree(pts, rng.choice([1, 2, 4, 8]),
                               rng.choice([0, 1, 2, 5, 8]))
    for _ in range(30):
        center = tuple(rng.uniform(-0.2, 1.2) for _ in range(3))
        r = rng.uniform(0.0, 0.6)
        hits = octree.query_sphere(root, center, r)
        assert _sorted(hits) == _sorted(
            octree.brute_force_sphere(pts, center, r))
        assert len(hits) == len(set(hits)) or len(pts) != len(set(pts))


@pytest.mark.parametrize("seed", range(8))
def test_box_matches_brute_force(seed):
    rng = random.Random(100 + seed)
    pts = _rand_points(rng, 120, grid=(seed % 2 == 0))
    root = octree.build_octree(pts, rng.choice([1, 2, 4, 8]),
                               rng.choice([0, 1, 2, 5, 8]))
    for _ in range(30):
        lo = tuple(rng.uniform(-0.2, 1.2) for _ in range(3))
        hi = tuple(v + rng.uniform(0.0, 0.5) for v in lo)
        box = octree.AABB(lo, hi)
        hits = octree.query_box(root, box)
        assert _sorted(hits) == _sorted(octree.brute_force_box(pts, box))


def test_query_box_accepts_plain_tuple_pair():
    root = octree.build_octree(_pts(10), 2, 4)
    box = ((0.1, 0.1, 0.1), (0.8, 0.8, 0.8))
    assert _sorted(octree.query_box(root, box)) == _sorted(
        octree.brute_force_box(_pts(10), octree.AABB(*box)))


def test_queries_report_candidate_nodes_without_full_scan():
    rng = random.Random(0)
    pts = _rand_points(rng, 400)
    root = octree.build_octree(pts, 4, 8)
    stats = octree.tree_stats(root)
    hits = octree.query_sphere(root, (0.5, 0.5, 0.5), 0.02)
    assert hits.candidate_nodes > 0
    assert hits.candidate_nodes <= stats["nodes"]
    assert hits.nodes_visited <= stats["nodes"]
    assert hits.points_tested < len(pts), "不应退化为全表扫描"
    assert len(hits) == len(octree.brute_force_sphere(
        pts, (0.5, 0.5, 0.5), 0.02))


def test_query_stats_dict_helper():
    root = octree.build_octree(_pts(20), 2, 5)
    payload = octree.query_sphere_stats(root, (0.5, 0.5, 0.5), 0.2)
    assert set(payload) == {"hits", "candidate_nodes", "points_tested",
                            "nodes_visited"}
    assert _sorted(payload["hits"]) == _sorted(
        octree.query_sphere(root, (0.5, 0.5, 0.5), 0.2))


# --------------------------------------------------------------------------- #
# 3. 边界与退化场景
# --------------------------------------------------------------------------- #

def test_points_exactly_on_split_boundary_have_one_home():
    # 全部点落在根盒中心面上：每个点只能被一个叶子接收。
    pts = [(0.5, 0.5, 0.5)] * 5 + [(0.5, 0.5, 0.0), (0.5, 0.0, 0.5),
                                   (0.0, 0.5, 0.5), (1.0, 1.0, 1.0)]
    root = octree.build_octree(pts, 1, 6)
    assert _sorted(_all_points(root)) == _sorted(pts)
    assert _sorted(octree.query_box(
        root, octree.AABB((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))) == \
        [(0.5, 0.5, 0.5)] * 5


def test_sphere_surface_and_box_face_are_inclusive():
    pts = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    root = octree.build_octree(pts, 1, 4)
    assert _sorted(octree.query_sphere(root, (0, 0, 0), 1.0)) == _sorted(pts)
    box = octree.AABB((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert _sorted(octree.query_box(root, box)) == _sorted(pts)


def test_zero_radius_and_negative_radius():
    pts = [(0.2, 0.3, 0.4), (0.2, 0.3, 0.4), (0.9, 0.9, 0.9)]
    root = octree.build_octree(pts, 1, 4)
    assert _sorted(octree.query_sphere(root, (0.2, 0.3, 0.4), 0.0)) == \
        [(0.2, 0.3, 0.4)] * 2
    assert octree.query_sphere(root, (0.2, 0.3, 0.4), -1.0) == []
    assert octree.brute_force_sphere(pts, (0.2, 0.3, 0.4), -1.0) == []


def test_duplicate_points_are_all_returned():
    pts = [(0.5, 0.5, 0.5)] * 17 + [(0.0, 0.0, 0.0)]
    root = octree.build_octree(pts, 2, 8)
    octree.validate_tree(root)
    hits = octree.query_sphere(root, (0.5, 0.5, 0.5), 0.001)
    assert len(hits) == 17
    assert _sorted(hits) == _sorted(octree.brute_force_sphere(
        pts, (0.5, 0.5, 0.5), 0.001))


def test_all_identical_points_do_not_split_forever():
    pts = [(0.5, 0.5, 0.5)] * 50
    root = octree.build_octree(pts, 2, 8)
    assert octree.tree_stats(root)["nodes"] == 1
    assert len(octree.query_sphere(root, (0.5, 0.5, 0.5), 0.0)) == 50


def test_degenerate_axis_root_box():
    pts = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)]
    root = octree.build_octree(pts, 1, 6)
    octree.validate_tree(root)
    assert _sorted(_all_points(root)) == _sorted(pts)


def test_empty_degenerate_queries():
    empty = octree.build_octree([], 4, 6)
    assert octree.query_box(empty, octree.AABB((0, 0, 0), (1, 1, 1))) == []
    assert octree.query_sphere(empty, (0, 0, 0), 0.0) == []
    assert octree.query_sphere(
        empty, (0, 0, 0), 0.0).candidate_nodes == 0
    assert octree.tree_stats(empty) == {
        "nodes": 1, "depth": 1, "leaves": 1, "points": 0}
    assert octree.query_many(empty, [((0, 0, 0), 1.0)])[0]["hits"] == []


def test_inverted_box_returns_nothing():
    root = octree.build_octree(_pts(10), 2, 4)
    assert octree.query_box(root, octree.AABB((1, 1, 1), (0, 0, 0))) == []
    assert octree.brute_force_box(_pts(10),
                                 octree.AABB((1, 1, 1), (0, 0, 0))) == []


def test_single_point_tree():
    root = octree.build_octree([(3.0, -2.0, 0.5)], 4, 6)
    assert len(octree.query_sphere(root, (3.0, -2.0, 0.5), 0.0)) == 1
    assert octree.query_sphere(root, (0.0, 0.0, 0.0), 1.0) == []


# --------------------------------------------------------------------------- #
# 4. 输出稳定性
# --------------------------------------------------------------------------- #

def test_two_runs_give_identical_tree_and_query(tmp_path):
    pts = _pts(30)
    runs = []
    for i in range(2):
        root = octree.build_octree(pts, 3, 6)
        out = tmp_path / ("run%d.json" % i)
        payload = {
            "tree": octree.serialize_tree(root),
            "sphere": [list(p) for p in
                       octree.query_sphere(root, (0.5, 0.5, 0.5), 0.4)],
            "box": [list(p) for p in
                    octree.query_box(root, octree.AABB(
                        (0.1, 0.1, 0.1), (0.7, 0.7, 0.7)))],
        }
        out.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False),
                       encoding="utf-8")
        runs.append(out.read_text(encoding="utf-8"))
    assert runs[0] == runs[1]


def test_cli_writes_consistent_output(tmp_path):
    import octree.cli as cli
    out1 = tmp_path / "a.json"
    out2 = tmp_path / "b.json"
    for out in (out1, out2):
        cli.main(["data/points.json", "--radius", "2.5", "--out", str(out),
                  "--batch"])
    a = json.loads(out1.read_text(encoding="utf-8"))
    b = json.loads(out2.read_text(encoding="utf-8"))
    assert a == b
    assert a["stats"]["missing_vs_brute_force"] == []
    assert a["stats"]["extra_vs_brute_force"] == []
    assert a["reproducibility"]["build_twice_identical"] is True
    assert a["reproducibility"]["query_twice_identical"] is True


# --------------------------------------------------------------------------- #
# 5. 动态维护
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", range(6))
def test_remove_equals_rebuild_and_never_inflates_depth(seed):
    rng = random.Random(200 + seed)
    pts = _rand_points(rng, 60, grid=(seed % 2 == 0))
    cap = rng.choice([1, 2, 4])
    depth = rng.choice([2, 4, 8])
    root = octree.build_octree(pts, cap, depth)
    reference = list(pts)
    order = list(pts)
    rng.shuffle(order)
    deepest = octree.tree_stats(root)["depth"]
    for p in order:
        root = octree.remove_point(root, p)
        reference.remove(p)
        octree.validate_tree(root)
        stats = octree.tree_stats(root)
        assert stats["depth"] <= deepest, "删除不得让树深膨胀"
        rebuilt = octree.build_octree(reference, cap, depth)
        assert octree.canonical_tree(root) == octree.canonical_tree(rebuilt)
    assert octree.tree_stats(root) == {
        "nodes": 1, "depth": 1, "leaves": 1, "points": 0}


def test_remove_reclaims_empty_nodes():
    pts = [(0.1, 0.1, 0.1), (0.9, 0.9, 0.9), (0.2, 0.8, 0.2), (0.8, 0.2, 0.8)]
    root = octree.build_octree(pts, 1, 6)
    before = octree.tree_stats(root)["nodes"]
    for p in pts[:3]:
        root = octree.remove_point(root, p)
        octree.validate_tree(root)  # 校验器禁止空叶子与非必需内部节点
    after = octree.tree_stats(root)
    assert after["nodes"] < before
    assert after["nodes"] == 1
    assert after["points"] == 1


def test_remove_missing_point_is_noop():
    root = octree.build_octree(_pts(15), 2, 5)
    before = octree.canonical_tree(root)
    assert octree.remove_point(root, (9.0, 9.0, 9.0)) is root
    assert octree.canonical_tree(root) == before


def test_remove_one_duplicate_at_a_time():
    pts = [(0.5, 0.5, 0.5)] * 4 + [(0.1, 0.1, 0.1)]
    root = octree.build_octree(pts, 1, 6)
    for remaining in (3, 2, 1, 0):
        root = octree.remove_point(root, (0.5, 0.5, 0.5))
        assert _sorted(_all_points(root)).count((0.5, 0.5, 0.5)) == remaining
        assert _sorted(_all_points(root)) == _sorted(
            [(0.5, 0.5, 0.5)] * remaining + [(0.1, 0.1, 0.1)])


def test_remove_until_empty_then_reinsert():
    pts = _pts(10)
    root = octree.build_octree(pts, 1, 5)
    for p in pts:
        root = octree.remove_point(root, p)
    assert octree.query_sphere(root, (0.5, 0.5, 0.5), 10.0) == []
    for p in pts:
        root = octree.insert_point(root, p)
    assert octree.canonical_tree(root) == octree.canonical_tree(
        octree.build_octree(pts, 1, 5))


@pytest.mark.parametrize("seed", range(6))
def test_insert_equals_full_rebuild(seed):
    rng = random.Random(300 + seed)
    base = _rand_points(rng, 25, grid=(seed % 2 == 0))
    extra = _rand_points(rng, 15, grid=(seed % 3 == 0))
    cap, depth = rng.choice([1, 2, 4]), rng.choice([1, 3, 6])
    root = octree.build_octree(base, cap, depth)
    reference = list(base)
    for p in extra:
        root = octree.insert_point(root, p)
        reference.append(p)
        octree.validate_tree(root)
        assert octree.canonical_tree(root) == octree.canonical_tree(
            octree.build_octree(reference, cap, depth))
    # 插入后查询仍与暴力一致
    for q in [((0.5, 0.5, 0.5), 0.3), ((0.0, 0.0, 0.0), 1.5)]:
        assert _sorted(octree.query_sphere(root, *q)) == _sorted(
            octree.brute_force_sphere(reference, *q))


def test_insert_out_of_bounds_grows_root_box():
    root = octree.build_octree(_pts(10), 2, 5)
    old_box = root.box
    root = octree.insert_point(root, (50.0, 50.0, 50.0))
    assert root.box.lo[0] <= min(p[0] for p in _all_points(root))
    assert root.box.hi[0] >= 50.0
    assert root.box != old_box
    assert _sorted(octree.query_sphere(root, (50.0, 50.0, 50.0), 0.0)) == \
        [(50.0, 50.0, 50.0)]


def test_insert_into_empty_tree():
    root = octree.build_octree([], 2, 5)
    root = octree.insert_point(root, (0.25, 0.5, 0.75))
    assert _sorted(_all_points(root)) == [(0.25, 0.5, 0.75)]
    assert _sorted(octree.query_box(
        root, octree.AABB((0, 0, 0), (1, 1, 1)))) == [(0.25, 0.5, 0.75)]


def test_insert_many_identical_points_respects_capacity_and_depth():
    root = octree.build_octree([], 2, 3)
    for _ in range(30):
        root = octree.insert_point(root, (0.5, 0.5, 0.5))
    octree.validate_tree(root)
    assert len(octree.query_sphere(root, (0.5, 0.5, 0.5), 0.0)) == 30


def test_limited_tree_insert_and_remove_keep_budget_and_queries_exact():
    pts = _pts(30)
    root = octree.build_limited(pts, 1, 10, 7)
    root = octree.insert_point(root, (0.123, 0.456, 0.789))
    assert octree.tree_stats(root)["nodes"] <= 7
    root = octree.remove_point(root, pts[0])
    assert octree.tree_stats(root)["nodes"] <= 7
    reference = pts[1:] + [(0.123, 0.456, 0.789)]
    for q in [((0.5, 0.5, 0.5), 0.4), ((0.1, 0.1, 0.1), 0.2)]:
        assert _sorted(octree.query_sphere(root, *q)) == _sorted(
            octree.brute_force_sphere(reference, *q))


# --------------------------------------------------------------------------- #
# 5b. 节点预算
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("max_nodes", [1, 2, 3, 5, 8, 13])
def test_build_limited_never_exceeds_budget(max_nodes):
    rng = random.Random(max_nodes)
    pts = _rand_points(rng, 80)
    root = octree.build_limited(pts, 1, 20, max_nodes)
    octree.validate_tree(root)
    assert octree.tree_stats(root)["nodes"] <= max_nodes
    for _ in range(10):
        center = tuple(rng.random() for _ in range(3))
        r = rng.uniform(0.0, 0.4)
        assert _sorted(octree.query_sphere(root, center, r)) == _sorted(
            octree.brute_force_sphere(pts, center, r))
        lo = tuple(rng.random() for _ in range(3))
        box = octree.AABB(lo, tuple(v + 0.3 for v in lo))
        assert _sorted(octree.query_box(root, box)) == _sorted(
            octree.brute_force_box(pts, box))


def test_budget_exhaustion_stops_splitting_but_keeps_points():
    pts = _rand_points(random.Random(1), 100)
    root = octree.build_limited(pts, 1, 20, 1)
    assert octree.tree_stats(root)["nodes"] == 1
    assert len(root.points) == 100
    assert _sorted(octree.query_sphere(root, (0.5, 0.5, 0.5), 10.0)) == \
        _sorted(pts)


def test_larger_budget_never_worse_than_smaller():
    pts = _rand_points(random.Random(2), 150)
    small = octree.build_limited(pts, 2, 12, 4)
    large = octree.build_limited(pts, 2, 12, 40)
    assert octree.tree_stats(large)["nodes"] >= octree.tree_stats(small)["nodes"]
    box = octree.AABB((0.2, 0.2, 0.2), (0.8, 0.8, 0.8))
    assert _sorted(octree.query_box(small, box)) == \
        _sorted(octree.query_box(large, box)) == \
        _sorted(octree.brute_force_box(pts, box))


def test_limited_tree_serializes_and_roundtrips():
    pts = _rand_points(random.Random(3), 40)
    root = octree.build_limited(pts, 1, 10, 6)
    restored = octree.tree_from_json(octree.tree_to_json(root))
    assert octree.canonical_tree(restored) == octree.canonical_tree(root)
    octree.validate_tree(restored)
    assert octree.tree_stats(restored)["nodes"] <= 6


# --------------------------------------------------------------------------- #
# 6. 批量查询
# --------------------------------------------------------------------------- #

def test_query_many_matches_individual_and_reports_savings():
    rng = random.Random(4)
    pts = _rand_points(rng, 300)
    root = octree.build_octree(pts, 4, 8)
    queries = [(tuple(rng.random() for _ in range(3)), rng.uniform(0.0, 0.1))
               for _ in range(12)]
    many = octree.query_many(root, queries)
    assert len(many) == len(queries)
    total = octree.tree_stats(root)["nodes"]
    for (center, r), row in zip(queries, many):
        single = octree.query_sphere(root, center, r)
        assert row["hits"] == [tuple(p) for p in single]
        assert row["hits"] == octree.brute_force_sphere(pts, center, r) or \
            _sorted(row["hits"]) == _sorted(
                octree.brute_force_sphere(pts, center, r))
        assert row["total"] == total
        assert row["visited"] == single.nodes_visited <= total
        assert row["saved"] == total - row["visited"]
        assert 0.0 <= row["saved_ratio"] <= 1.0
        assert row["saved_ratio"] == pytest.approx(
            1.0 - row["visited"] / total)
        assert row["points_tested"] < len(pts)


def test_query_many_empty_and_degenerate():
    root = octree.build_octree(_pts(5), 2, 4)
    assert octree.query_many(root, []) == []
    row = octree.query_many(root, [((0.5, 0.5, 0.5), -1.0)])[0]
    assert row["hits"] == []
    assert row["visited"] <= row["total"]


def test_query_many_batch_is_cheaper_than_full_scan_on_average():
    rng = random.Random(5)
    pts = _rand_points(rng, 500)
    root = octree.build_octree(pts, 4, 8)
    queries = [(tuple(rng.random() for _ in range(3)), 0.03) for _ in range(20)]
    rows = octree.query_many(root, queries)
    assert sum(r["visited"] for r in rows) < sum(r["total"] for r in rows)
    assert sum(r["saved_ratio"] for r in rows) / len(rows) > 0.5


# --------------------------------------------------------------------------- #
# 结构校验器与数据文件保护
# --------------------------------------------------------------------------- #

def test_validate_tree_rejects_broken_structures():
    root = octree.build_octree(_pts(20), 1, 5)
    octree.validate_tree(root)
    # 父节点持有叶子数据
    bad = octree.tree_from_json(octree.tree_to_json(root))
    internal = next(n for n in _walk(bad) if n.children)
    internal.points.append((0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        octree.validate_tree(bad)
    # 空叶子未回收
    bad2 = octree.tree_from_json(octree.tree_to_json(root))
    internal = next(n for n in _walk(bad2) if n.children)
    empty = octree.Node(octree.AABB((0, 0, 0), (0, 0, 0)), internal.depth + 1)
    internal.children.append(empty)
    with pytest.raises(ValueError):
        octree.validate_tree(bad2)


def test_data_file_is_untouched():
    path = "data/points.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data == [[0, 0, 0], [1, 1, 1], [-1, 0, 2], [5, 5, 5],
                    [0, 0, 0], [-1, 0, 2]]
