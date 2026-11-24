"""大名册性能冒烟：每区 5000 株，验证概率计算不枚举首轮名单。"""
import json
import time
import urllib.request


def roster(prefix, n, slope):
    return [{"id": f"{prefix}-{i}", "is_slope": i < slope} for i in range(n)]


payload = {
    "districts": [
        {
            "name": "甲",
            "roster": roster("A", 5000, 2000),
            "first_round_quota": 300,
            "actual_drawn": [f"A-{i}" for i in range(300)],
        },
        {
            "name": "乙",
            "roster": roster("B", 5000, 1000),
            "first_round_quota": 1200,
            "actual_drawn": [f"B-{i}" for i in range(1200)],
        },
    ],
    "supplement_quota": 800,
    "seed": 1,
}
t = time.time()
req = urllib.request.Request(
    "http://127.0.0.1:8000/sample",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
r = json.load(urllib.request.urlopen(req))
print(
    f"耗时 {time.time() - t:.3f}s, 入样 {len(r['entries'])} 株, "
    f"补抽池 {r['supplement_pool_size']}, 补抽 {r['supplement_drawn']}"
)
print("坡地π 甲/乙:", r["inclusion_probabilities"]["甲"]["slope"],
      r["inclusion_probabilities"]["乙"]["slope"])
print("非坡地π 甲/乙:", r["inclusion_probabilities"]["甲"]["non_slope"],
      r["inclusion_probabilities"]["乙"]["non_slope"])
