# -*- coding: utf-8 -*-
"""八叉树构建与区域查询"""

from dataclasses import dataclass, field


@dataclass
class AABB:
    """轴对齐包围盒。"""
    lo: tuple
    hi: tuple


@dataclass
class Node:
    """八叉树节点。"""
    box: AABB
    depth: int = 0
    children: list = field(default_factory=list)
    points: list = field(default_factory=list)


def build_octree(points, capacity, max_depth):
    """建树"""
    raise NotImplementedError("build_octree 尚未实现")


def query_sphere(root, center, r):
    """半径查询"""
    raise NotImplementedError("query_sphere 尚未实现")


def query_box(root, box):
    """盒子查询"""
    raise NotImplementedError("query_box 尚未实现")

