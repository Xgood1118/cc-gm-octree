# 八叉树构建与区域查询

一个能构建八叉树、支持半径与盒子查询的索引器，仅依赖 Python 标准库。

## 保留签名

- `Node`
- `AABB`
- `build_octree(points, capacity, max_depth)`
- `query_sphere(root, center, r)`
- `query_box(root, box)`

## 设计要点

### 构建

- `capacity`（叶子容量）与 `max_depth`（最大深度，根为 0）均可配置；
  叶子点数超过容量且未达最大深度时**必须分裂**为 8 个卦限子节点。
- 分裂后父节点 `points` 立即清空——**内部节点不持有任何叶子数据**，
  每个点恰好归属一个叶子，不会重复。
- 点落在分裂面上时归入正侧卦限（`p >= center`），边界点归属确定。
- 建树前对输入点按坐标排序，点的归属只由坐标与盒边界决定，
  **与输入顺序无关**：同一批点连跑两次（即使打乱顺序）得到字节级一致的
  序列化结构。
- 到达 `max_depth` 后叶子不再分裂（重复点等无法再分的情形有确定结果）；
  空树生成以原点为中心的单位根盒，查询安全返回空。

### 查询

- `query_sphere` / `query_box` 只下钻 AABB 与查询区域相交的节点，
  不相交的整支直接剪枝。
- 返回 `QueryResult`（`list` 子类，可直接当命中点列表使用），额外携带：
  - `candidate_nodes`：与查询区域相交、实际访问的**候选节点数**；
  - `points_tested`：在候选叶子中真正做几何判定的点数（小于总点数即证明没有全表扫描）；
  - `nodes_visited`：遍历访问的全部节点数（含被剪枝节点）。
- 球面与盒面均按闭区间判定（`<=`），边界点命中确定；重复点逐份保留、逐份命中。
- 每个点只存于一个叶子，因此查询无遗漏也无重复命中。

### 序列化

- `dumps_tree(root)` / `loads_tree(text)`：纯 JSON 的结构序列化与读回，
  键序固定，往返后再次序列化字节级一致，读回的树查询结果与原树相同。
- 另有 `tree_to_dict` / `tree_from_dict`、`count_nodes` / `tree_depth`、
  `brute_force_sphere` / `brute_force_box`（暴力对照）辅助函数。

## 实测结果

命令：

```
python -m octree.cli data/points.json --radius 2.5 --out hits.json
```

数据：`data/points.json` 共 6 个点（含重复点 `[0,0,0]`、`[-1,0,2]`），
建树参数 `capacity=4`、`max_depth=6`，查询中心 `(0, 0, 0)`、半径 `2.5`。

| 指标 | 值 |
| --- | --- |
| 节点数 | **6** |
| 最大深度 | **2** |
| 候选节点数 | **5** |
| 实测点数 / 总点数 | **5 / 6**（未全表扫描） |
| 命中点数 | **5** |
| 暴力比对命中数 | 5 |
| 与暴力结果差异 | **missing=0，extra=0（完全一致）** |

两次运行输出的 `hits.json` 字节级一致；随机化测试（200 组随机点集 ×
随机球/盒查询，含重复点与空树）查询结果均与逐点暴力比对一致。

盒子查询同样可用：

```
python -m octree.cli data/points.json --box -2 -2 -2 2 2 2 --out hits_box.json
```

## 自测

```
python -m pytest tests -q
python -m octree.cli data/points.json --radius 2.5 --out hits.json
```

测试覆盖：分裂后叶子无父持有、最大深度止裂、序列化往返（含空树）、
球/盒查询与暴力一致、候选节点剪枝证明、节点边界点、球面/盒面闭区间、
重复点逐份命中、空树、参数校验、重复构建确定性，共 17 个用例。
