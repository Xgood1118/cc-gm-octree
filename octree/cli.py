# -*- coding: utf-8 -*-
"""命令行入口：建树、查询，并与暴力结果对照。"""

import argparse
import json
from collections import Counter

import octree as m


def _multiset_diff(a, b):
    """两个命中列表（允许重复点）的多重集差异：(a 缺少的, a 多出的)。"""
    ca, cb = Counter(map(tuple, a)), Counter(map(tuple, b))
    missing = list((cb - ca).elements())
    extra = list((ca - cb).elements())
    return missing, extra


def main(argv=None):
    ap = argparse.ArgumentParser(description="八叉树构建与区域查询")
    ap.add_argument("points", help="点集 JSON 文件，如 data/points.json")
    ap.add_argument("--capacity", type=int, default=4, help="叶子容量")
    ap.add_argument("--max-depth", type=int, default=6, help="最大深度")
    ap.add_argument("--radius", type=float, default=2.5,
                    help="半径查询的半径（默认 2.5）")
    ap.add_argument("--center", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                    metavar=("X", "Y", "Z"), help="半径查询中心")
    ap.add_argument("--box", type=float, nargs=6,
                    metavar=("LO_X", "LO_Y", "LO_Z", "HI_X", "HI_Y", "HI_Z"),
                    help="改为盒子查询，给出盒子的 lo/hi 两个角点")
    ap.add_argument("--out", required=True, help="查询结果输出 JSON 路径")
    args = ap.parse_args(argv)

    pts = m.load_points(args.points)
    root = m.build_octree(pts, args.capacity, args.max_depth)

    if args.box is not None:
        lo = tuple(args.box[:3])
        hi = tuple(args.box[3:])
        box = m.AABB(lo, hi)
        result = m.query_box(root, box)
        brute = m.brute_force_box(pts, box)
        query_kind = "box"
        region = {"lo": list(lo), "hi": list(hi)}
    else:
        center = tuple(args.center)
        result = m.query_sphere(root, center, args.radius)
        brute = m.brute_force_sphere(pts, center, args.radius)
        query_kind = "sphere"
        region = {"center": list(center), "radius": args.radius}

    missing, extra = _multiset_diff(result, brute)
    payload = {
        "query": query_kind,
        "region": region,
        "stats": {
            "node_count": m.count_nodes(root),
            "max_depth": m.tree_depth(root),
            "candidate_nodes": result.candidate_nodes,
            "points_tested": result.points_tested,
            "total_points": len(pts),
            "hit_count": len(result),
            "brute_force_hit_count": len(brute),
            "missing_vs_brute_force": [list(p) for p in missing],
            "extra_vs_brute_force": [list(p) for p in extra],
        },
        "hits": [list(p) for p in result],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)

    print("query           : %s" % query_kind)
    print("nodes           : %d" % payload["stats"]["node_count"])
    print("max depth       : %d" % payload["stats"]["max_depth"])
    print("candidate nodes : %d" % result.candidate_nodes)
    print("points tested   : %d / %d"
          % (result.points_tested, len(pts)))
    print("hits            : %d (brute force: %d)"
          % (len(result), len(brute)))
    print("diff vs brute   : missing=%d extra=%d"
          % (len(missing), len(extra)))
    print("result written  : %s" % args.out)


if __name__ == "__main__":
    main()
