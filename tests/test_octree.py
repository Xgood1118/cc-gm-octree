# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import octree as mod  # noqa: E402

def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_hex(path):
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def test_build_and_query():
    pts = octree.load_points("data/points.json")
    root = octree.build_octree(pts, 4, 6)
    hits = octree.query_sphere(root, (0, 0, 0), 2.5)
    brute = [p for p in pts if sum((a - b) ** 2 for a, b in zip(p, (0, 0, 0))) <= 6.25]
    assert len(hits) == len(brute)


def test_empty_tree():
    root = octree.build_octree([], 4, 6)
    assert octree.query_sphere(root, (0, 0, 0), 1.0) == []
