# 八叉树构建与区域查询

一个纯标准库实现的八叉树索引器：构建、半径/盒子查询、序列化、动态插删、
节点预算与批量查询。保留 `Node`、`AABB`、`build_octree`、`query_sphere`、
`query_box` 的原始名字与签名。

## 保留签名

```python
Node(box, depth=0, children=[], points=[])
AABB(lo, hi)
build_octree(points, capacity, max_depth)
query_sphere(root, center, r)
query_box(root, box)
```

`query_sphere` / `query_box` 返回 `QueryResult`（`list` 的子类，可直接当列表用），
额外带有 `candidate_nodes`（与查询区域相交、被实际遍历的节点数）、
`points_tested`（真正做几何判定的点数）、`nodes_visited`（检查过包围盒的节点数，
含被剪枝节点）。

## 设计要点

- **盒子语义**：闭区间 `[lo, hi]`；点落在分裂中心面上时归入坐标 `>= center`
  的正侧卦限，每个点唯一属于一个叶子，不会重复命中。
- **分裂**：`点的数量 > capacity` 且 `depth < max_depth` 且该节点至少能分到
  两个卦限时才分裂。内部节点 `points` 必为空，点只存在于叶子里。
- **确定性**：点的归属只由坐标决定，与输入顺序无关；叶子内的点按字典序排序，
  子节点按卦限升序排列。同一批点、同一组参数两次构建得到逐字节相同的
  规范化 JSON。
- **序列化**：`serialize_tree` / `deserialize_tree`（及 `tree_to_json` /
  `tree_from_json`、`save_tree` / `load_tree`）保存完整结构与构建参数，
  读回后可继续插入、删除。
- **删除**：从叶子删点后沿路径自底向上回收空子节点、必要时折叠；若剩余点集的
  根包围盒发生变化（删掉了极值点），退化为按剩余点全量重建。删除后的树与
  “用剩余点全量重建”在 `canonical_tree(root)` 下完全一致，树深不会增加。
- **节点预算**：`build_limited` 采用广度优先分裂，预算不足时跳过该次分裂
  （节点保留为叶子并保留全部点）。节点数严格不超过 `max_nodes`，查询仍逐点
  判定，因此结果与暴力枚举一致。
- **批量查询**：`query_many(root, queries)` 返回逐条
  `{hits, visited, total, candidate_nodes, points_tested, saved, saved_ratio,
  visited_ratio}`，`visited` 与逐个 `query_sphere` 完全一致，
  `saved_ratio = 1 - visited / total`。

## 自测

```
python -m pytest tests -q
```

当前结果：`79 passed`。

覆盖内容：超容量必分裂、父节点不持有叶子数据、最大深度与容量可配置、
构建顺序无关与两次构建同构、序列化往返（含文件）、查询与暴力逐点对照
（球/盒各 8 组随机种子）、候选节点数小于全表、边界点/重复点/空树/单点/
退化轴/零半径/负半径/反向盒、两次运行输出一致、删除与全量重建规范等价
且树深不膨胀、空节点回收、删除到空后重新插入、插入与全量重建等价、
越界插入扩展根盒、预算上限与预算耗尽、批量查询与逐个一致及节省比例、
结构校验器拒绝非法树。

## CLI 验收

```
python -m octree.cli data/points.json --radius 2.5 --out hits.json
python -m octree.cli data/points.json --box -2 -2 -2 2 2 2 --out hits_box.json
python -m octree.cli --generate 20000 --capacity 8 --max-depth 8 \
    --radius 0.05 --center 0.5 0.5 0.5 --batch --out big.json
python -m octree.cli --generate 20000 --capacity 8 --max-depth 8 \
    --max-nodes 200 --radius 0.05 --center 0.5 0.5 0.5 --out lim.json
```

### 1) `data/points.json`，半径 2.5

| 指标 | 数值 |
| --- | --- |
| 节点数 / 叶子数 | 6 / 4 |
| 最大深度 | 2 |
| 候选节点数 | 5 |
| 实际判定点数 | 5 / 6 |
| 命中点数 | 5（暴力 5） |
| 与暴力差异 | 缺失 0，多余 0 |
| 两次构建 / 两次查询一致 | True / True |
| 规范化树 SHA-256 | `1cb98622058473c33b91f936becb0590c373df6f6cdc8e7a5b9f1cb8c6ece52a` |

盒子查询 `[-2,-2,-2] .. [2,2,2]` 得到完全相同的数值：6 个节点、最大深度 2、
候选节点 5、5 / 6 点判定、命中 5、缺失 0、多余 0。

> 这组数据只有 6 个点且查询范围覆盖全树，候选节点数接近全量是正常的；
> 下面的 2 万点演示才体现索引收益。

### 2) 确定性生成的 20000 点，容量 8、最大深度 8、半径 0.05

| 指标 | 数值 |
| --- | --- |
| 节点数 / 叶子数 | 6074 / 5243 |
| 最大深度 | 5 |
| 候选节点数 | 33 |
| 访问节点数 / 总节点数 | 201 / 6074（节省 96.7%） |
| 实际判定点数 | 37 / 20000 |
| 命中点数 | 9（暴力 9） |
| 与暴力差异 | 缺失 0，多余 0 |
| 批量球查询平均节省 | 98.8%（5 次查询：62、65、64、113、57 个访问节点 / 6074） |

### 3) 同一批 20000 点，节点上限 200

| 指标 | 数值 |
| --- | --- |
| 节点数 / 叶子数 | 200 / 175（上限 200） |
| 最大深度 | 4 |
| 候选节点数 | 19 |
| 访问节点数 / 总节点数 | 89 / 200（节省 55.5%） |
| 实际判定点数 | 1977 / 20000 |
| 命中点数 | 9（暴力 9） |
| 与暴力差异 | 缺失 0，多余 0 |

预算耗尽后停止分裂的叶子保留全部点，查询逐点判定，结果仍与暴力完全一致。
