# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import octree as mod  # noqa: E402


def test_build_and_query():
    pts = octree.load_points("data/points.json")
    root = octree.build_octree(pts, 4, 6)
    hits = octree.query_sphere(root, (0, 0, 0), 2.5)
    brute = [p for p in pts if sum((a - b) ** 2 for a, b in zip(p, (0, 0, 0))) <= 6.25]
    assert len(hits) == len(brute)


def test_empty_tree():
    root = octree.build_octree([], 4, 6)
    assert octree.query_sphere(root, (0, 0, 0), 1.0) == []


# ===== v2 契约测试：初始全部失败，实现后必须通过 =====
import octree  # noqa: E402



def _pts(n=12):
    return [((i * 7 % 13) / 13.0, (i * 5 % 11) / 11.0, (i * 3 % 7) / 7.0) for i in range(n)]


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
    assert sorted(octree.query_sphere(root, (0.5, 0.5, 0.5), 0.5)) == sorted(
        octree.brute_force_sphere(pts, (0.5, 0.5, 0.5), 0.5))


def test_query_many_matches_single():
    pts = _pts(10)
    root = octree.build_limited(pts, 2, 8, 64)
    qs = [((0.2, 0.2, 0.2), 0.2), ((0.8, 0.8, 0.8), 0.3)]
    many = octree.query_many(root, qs)
    for q, r in zip(qs, many):
        assert sorted(r["hits"]) == sorted(octree.query_sphere(root, q[0], q[1]))
        assert r["visited"] <= r["total"]
