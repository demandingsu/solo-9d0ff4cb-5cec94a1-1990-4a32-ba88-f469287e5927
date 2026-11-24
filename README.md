# 两区坡地苗木两阶段抽样服务

园林调查员补抽坡地苗木的带权名单生成服务。两个区首轮各自独立、等概率
无放回抽样；首轮未入样的坡地苗木合并为补抽池，等概率无放回补抽，池内
不足补抽名额时全取。返回每株入样苗木的编号、轮次、整套方案下的入样
概率及其倒数（权重），供成活率分析使用。

## 目录结构

```
app/
  main.py        FastAPI 路由与异常处理
  sampling.py    抽样、校验、入样概率/权重计算（核心逻辑）
  schemas.py     请求/响应 Pydantic 模型
tests/
  test_sampling.py  超几何单测、全枚举精确核对、Monte Carlo 仿真、边界与 HTTP 测试
demo/
  request_*.json    三类可直接在 /docs 试用的请求样例
  perf_check.py     大名册性能冒烟脚本
Dockerfile, docker-compose.yml, requirements*.txt
```

## 启动方式（Docker Compose）

```bash
docker compose up --build
```

启动后访问：

- 交互文档（Swagger）：http://localhost:8000/docs
- OpenAPI Schema：http://localhost:8000/openapi.json
- 健康检查：http://localhost:8000/health
- 抽样接口：`POST /sample`

在 `/docs` 展开 `POST /sample` → Try it out，可粘贴 `demo/request_normal.json`
（常规）、`request_shortfall.json`（补抽不足全取）、
`request_impossible.json`（存在不可能入样株，被拒绝）的内容试用。

本地无 Docker 时也可直接运行：

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
.venv/bin/python -m pytest tests/ -q
```

## 请求与响应

请求字段：

- `districts`：恰好两个区；每区含
  - `name` 区名（两区须不同）
  - `roster` 完整名册，每株 `{id 唯一编号, is_slope 是否坡地}`
  - `first_round_quota` 首轮名额（可为 0，不能超过名册量）
  - `actual_drawn` 首轮实抽编号（须无重复、属于本区、数量等于名额）
- `supplement_quota`：事先定下的补抽名额（可为 0）
- `seed`：随机种子（整数）；相同输入重跑结果完全一致

响应字段：

- `entries`：入样名单，每条含 `id`、`district`、`round_name`（首轮/补抽）、
  `inclusion_probability`（整套方案下至少一次入样概率）、`weight`（概率倒数）
- `supplement_pool_size` / `supplement_drawn` / `supplement_quota` /
  `supplement_all_taken`：补抽池规模、实抽数、名额、是否全取
- `inclusion_probabilities`：每区非坡地、坡地两类苗木的入样概率

## 入样概率与权重的计算方法

记第 i 区名册总量 `N_i`、坡地数 `a_i`、首轮名额 `n_i`，首轮抽样比
`p_i = n_i / N_i`。

- **非坡地**只能通过首轮入样，`π = p_i`。
- **坡地** j：首轮未中（概率 `1-p_i`）后进入补抽池。给定池大小 S，
  补抽 `min(m,S)` 株的条件入样概率为 `g(S)=min(m,S)/S`。
  在“j 首轮未中”条件下，j 必在池中：
  - 本区其余 `N_i-1` 株（坡地 `a_i-1` 株）中抽 `n_i` 株，抽中坡地数
    `Z_i ~ Hypergeom(N_i-1, a_i-1, n_i)`（条件超几何）；
  - 另一区首轮抽中坡地数 `X_k ~ Hypergeom(N_k, a_k, n_k)`，两者独立；
  - 池大小（含 j）`S = 1 + (a_i-1-Z_i) + (a_k-X_k)`。

  补抽入样概率 `q_i = E[g(S)] = Σ_z Σ_x P(Z_i=z)P(X_k=x) g(S)`，
  坡地总入样概率 `π = 1 - (1-p_i)(1-q_i)`，权重 `w = 1/π`。

计算通过超几何 PMF 对首轮抽中坡地数的取值范围做双重求和
（规模 O(n_i·n_k)），**汇总首轮所有可能结果，不逐份枚举首轮名单**。
PMF 用对数组合数定起点 + 相邻项比值递推 + 归一化，兼顾大名册下的
数值稳定与速度；结果在 0/1 边界做 1e-12 吸附，消除浮点尾差。

## 校验与拒绝规则

- 首轮实抽编号：重复、不属于本区、数量与名额不符 → 422 拒绝；
- 名册编号区内/跨区重复、首轮名额超名册量、空名册、负名额等 → 422；
- **不可能入样检查**：若某区首轮名额为 0，则该区非坡地苗木永远无法
  入样；若两区名额与补抽名额均为 0，则所有苗木都无法入样。此时返回
  422，body 中 `impossible_ids` 列出具体编号并拒绝出单。

## 实测结果

测试与实测均在本仓库完成（`28 passed`），关键场景：

1. **两区抽样比例不同**（甲 2/10=0.2、乙 6/12=0.5）：Monte Carlo
   4 万次仿真频率与公式概率误差 < 0.02；同区同类型苗木概率一致。
2. **补抽不足全取**：池小于名额时全部入样，坡地 `π=1.0`、权重精确 `1.0`。
3. **全枚举精确核对**：小名册枚举首轮全部组合，入样概率与公式逐株一致
   （相对误差 1e-12）。
4. **确定性**：相同输入（含种子）两次 HTTP 响应逐字段相等；不同种子
   仅补抽结果可能变化。
5. **边界**：首轮名额为 0、补抽名额为 0、首轮全抽、区内无坡地均正确；
   不可能入样株按编号拒绝。
6. **大名册性能**：每区 5000 株（共 1 万株、首轮实抽 1500、补抽 800），
   纯计算约 0.3 秒，服务端处理（含 Pydantic 解析/序列化）约 0.46 秒。

## 未完成项 / 说明

- 当前运行环境未安装 Docker，Dockerfile 与 docker-compose.yml 已按
  常规约定编写并通过等价的本地 uvicorn 全量验证，但**未在本机实际
  执行 `docker compose up`**；在有 Docker 的环境执行该命令即可。
- 服务为无状态计算，未做鉴权、限流与持久化（需求未要求）。
- 随机数使用 Python 标准库 `random.Random(seed)`，跨 Python 版本的
  采样序列由标准库版本兼容性保证；若未来需要跨语言/跨版本严格一致，
  可改为固定的哈希或自研 PRNG。
