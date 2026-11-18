# 八叉树构建与区域查询

一个能构建八叉树、支持半径与盒子查询的索引器

## 保留签名

- `Node`
- `AABB`
- `build_octree(points, capacity, max_depth)`
- `query_sphere(root, center, r)`
- `query_box(root, box)`

## 自测

```
python -m pytest tests -q
python -m octree.cli data/points.json --radius 2.5 --out hits.json
```
