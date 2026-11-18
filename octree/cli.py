# -*- coding: utf-8 -*-
"""命令行入口：建树、查询，并与暴力结果对照。"""

import argparse
import hashlib
import json
import math
from collections import Counter

import octree as m


def _multiset_diff(a, b):
    """两个命中列表（允许重复点）的多重集差异：(b 缺少的, b 多出的)。"""
    ca, cb = Counter(map(tuple, a)), Counter(map(tuple, b))
    return list((cb - ca).elements()), list((ca - cb).elements())


def _generate(count, seed):
    """用确定性 LCG 生成 count 个 [0, 1)^3 点（仅用于演示规模效应）。"""
    mask = (1 << 64) - 1
    state = seed & mask
    values = []
    for _ in range(count * 3):
        state = (state * 6364136223846793005 + 1442695040888963407) & mask
        values.append((state >> 11) / float(1 << 53))
    return [tuple(values[i:i + 3]) for i in range(0, len(values), 3)]


def _build(points, args):
    if args.max_nodes is not None:
        return m.build_limited(points, args.capacity, args.max_depth,
                               args.max_nodes)
    return m.build_octree(points, args.capacity, args.max_depth)


def _batch(root, points, radius):
    """确定性地取若干查询中心，展示批量查询的节点节省。"""
    centers = sorted(set(points))
    if not centers:
        return []
    step = max(1, len(centers) // 5)
    picked = centers[::step][:5]
    queries = [(c, radius) for c in picked]
    return [dict({"center": list(c), "radius": r}, **row)
            for (c, r), row in zip(queries, m.query_many(root, queries))]


def main(argv=None):
    ap = argparse.ArgumentParser(description="八叉树构建与区域查询")
    ap.add_argument("points", nargs="?", help="点集 JSON 文件，如 data/points.json")
    ap.add_argument("--generate", type=int, metavar="N",
                    help="不读文件，改用确定性随机生成的 N 个点（演示用）")
    ap.add_argument("--seed", type=int, default=20261119, help="--generate 的随机种子")
    ap.add_argument("--capacity", type=int, default=4, help="叶子容量")
    ap.add_argument("--max-depth", type=int, default=6, help="最大深度")
    ap.add_argument("--max-nodes", type=int, default=None,
                    help="节点数上限；超出后停止分裂")
    ap.add_argument("--radius", type=float, default=2.5,
                    help="半径查询的半径（默认 2.5）")
    ap.add_argument("--center", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                    metavar=("X", "Y", "Z"), help="半径查询中心")
    ap.add_argument("--box", type=float, nargs=6,
                    metavar=("LO_X", "LO_Y", "LO_Z", "HI_X", "HI_Y", "HI_Z"),
                    help="改为盒子查询，给出盒子的 lo/hi 两个角点")
    ap.add_argument("--batch", action="store_true",
                    help="附带批量球查询的访问节点数与节省比例")
    ap.add_argument("--tree-out", help="把树结构（规范化 JSON）写到该路径")
    ap.add_argument("--out", required=True, help="查询结果输出 JSON 路径")
    args = ap.parse_args(argv)

    if args.generate is not None:
        pts = _generate(args.generate, args.seed)
    elif args.points:
        pts = m.load_points(args.points)
    else:
        ap.error("需要给出 points 文件或 --generate N")

    root = _build(pts, args)
    m.validate_tree(root)

    # 可复现性：同参数再建一次，树结构与查询结果都必须逐字节一致。
    root2 = _build(pts, args)
    canonical = m.canonical_tree(root)
    canonical2 = m.canonical_tree(root2)

    if args.box is not None:
        lo, hi = tuple(args.box[:3]), tuple(args.box[3:])
        query_box = m.AABB(lo, hi)
        result = m.query_box(root, query_box)
        result_again = m.query_box(root2, query_box)
        brute = m.brute_force_box(pts, query_box)
        query_kind = "box"
        region = {"lo": list(lo), "hi": list(hi)}
    else:
        center = tuple(args.center)
        result = m.query_sphere(root, center, args.radius)
        result_again = m.query_sphere(root2, center, args.radius)
        brute = m.brute_force_sphere(pts, center, args.radius)
        query_kind = "sphere"
        region = {"center": list(center), "radius": args.radius}

    missing, extra = _multiset_diff(result, brute)
    stats = m.tree_stats(root)
    payload = {
        "query": query_kind,
        "region": region,
        "params": {
            "capacity": args.capacity,
            "max_depth": args.max_depth,
            "max_nodes": args.max_nodes,
        },
        "stats": {
            "node_count": stats["nodes"],
            "leaf_count": stats["leaves"],
            "max_depth": m.tree_depth(root) - 1,
            "tree_levels": stats["depth"],
            "candidate_nodes": result.candidate_nodes,
            "visited_nodes": result.nodes_visited,
            "node_scan_saved_ratio": (
                1.0 - result.nodes_visited / stats["nodes"]
                if stats["nodes"] else 0.0),
            "points_tested": result.points_tested,
            "total_points": len(pts),
            "hit_count": len(result),
            "brute_force_hit_count": len(brute),
            "missing_vs_brute_force": [list(p) for p in missing],
            "extra_vs_brute_force": [list(p) for p in extra],
        },
        "reproducibility": {
            "build_twice_identical": canonical == canonical2,
            "query_twice_identical": list(result) == list(result_again),
            "tree_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        },
        "hits": [list(p) for p in result],
    }

    if args.batch:
        rows = _batch(root, pts, args.radius)
        payload["batch"] = rows
        if rows:
            payload["stats"]["batch_mean_saved_ratio"] = (
                sum(r["saved_ratio"] for r in rows) / len(rows))

    if args.tree_out:
        m.save_tree(args.tree_out, root)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)

    print("query           : %s" % query_kind)
    print("nodes / leaves  : %d / %d" % (stats["nodes"], stats["leaves"]))
    print("max depth       : %d" % (m.tree_depth(root) - 1))
    print("candidate nodes : %d (visited %d of %d, saved %.1f%%)"
          % (result.candidate_nodes, result.nodes_visited, stats["nodes"],
             100.0 * payload["stats"]["node_scan_saved_ratio"]))
    print("points tested   : %d / %d" % (result.points_tested, len(pts)))
    print("hits            : %d (brute force: %d)" % (len(result), len(brute)))
    print("diff vs brute   : missing=%d extra=%d" % (len(missing), len(extra)))
    if args.batch and payload.get("batch"):
        print("batch saved     : mean %.1f%% over %d queries"
              % (100.0 * payload["stats"]["batch_mean_saved_ratio"],
                 len(payload["batch"])))
    if not payload["reproducibility"]["build_twice_identical"]:
        print("WARNING         : 两次构建结果不一致")
    if not payload["reproducibility"]["query_twice_identical"]:
        print("WARNING         : 两次查询结果不一致")
    print("result written  : %s" % args.out)


if __name__ == "__main__":
    main()
