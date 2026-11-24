"""准备 Textual Inversion 训练数据：每个用户选最多 5 张 stage1 中评分 4–5 的图像。

选择规则（确定性）：
  1. 只保留 initial_rating ∈ {4, 5} 的 stage1 图像；
  2. 按评分降序，同分用 TI_SELECT_SEED + user_id + image_id 的哈希排序；
  3. 取前 MAX_REFS 张（不足 5 张则全取）。

输出：
  stylebench_ti/ti_data/<user_id>/img_XX.png   软链接（不复制大图）
  results/textual_inversion/ti_refs.csv        选择清单
"""

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict

import ti_config as C


def _h(*parts: str) -> str:
    return hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="rating45_hash", choices=["rating45_hash", "rating5_reverse"],
                    help="rating45_hash=评分4-5+哈希排序(默认,5图正式版); rating5_reverse=只取5分且按列表位置从后往前")
    ap.add_argument("--max-refs", type=int, default=None)
    ap.add_argument("--users", default=None, help="逗号分隔的用户子集；默认全部 58 人")
    ap.add_argument("--out-data-dir", default=None)
    ap.add_argument("--refs-csv", default=None)
    args = ap.parse_args()

    max_refs = args.max_refs if args.max_refs is not None else C.MAX_REFS
    data_dir = args.out_data_dir or C.DATA_DIR
    refs_path = args.refs_csv or os.path.join(os.path.dirname(C.RESULTS_JSONL), "ti_refs.csv")
    only = {u.strip() for u in args.users.split(",")} if args.users else None

    dataset_root = os.path.abspath(os.path.join(C.STYLEBENCH_DIR, "..", "dataset", "StylePreferenceBenchmark_389_v1"))
    users = json.load(open(os.path.join(dataset_root, "splits", "test.json"), encoding="utf-8"))["users"]
    images = {}
    with open(os.path.join(dataset_root, "metadata", "images.jsonl"), encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            images[rec["image_id"]] = rec

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.dirname(refs_path), exist_ok=True)

    rows = []
    stats = defaultdict(int)
    for user in users:
        uid = user["user_id"]
        if only is not None and uid not in only:
            continue
        if args.mode == "rating5_reverse":
            # 只取评分恰为 5 的图，按其在 stage1 列表中的位置从后往前取
            pool = [(i, r) for i, r in enumerate(user["stage1"]) if r.get("initial_rating") == 5]
            pool.sort(key=lambda x: -x[0])
            chosen = [r for _, r in pool[:max_refs]]
        else:
            pool = [r for r in user["stage1"] if r.get("initial_rating") in C.ALLOWED_RATINGS]
            pool.sort(key=lambda r: (-int(r["initial_rating"]), _h(str(C.TI_SELECT_SEED), uid, r["image_id"])))
            chosen = pool[:max_refs]

        user_dir = os.path.join(data_dir, uid)
        os.makedirs(user_dir, exist_ok=True)
        for old in os.listdir(user_dir):
            os.remove(os.path.join(user_dir, old))

        for idx, r in enumerate(chosen, 1):
            src = os.path.join(dataset_root, images[r["image_id"]]["file_name"])
            dst = os.path.join(user_dir, f"img_{idx:02d}.png")
            os.symlink(src, dst)
            rows.append(
                {
                    "user_id": uid,
                    "index": idx,
                    "image_id": r["image_id"],
                    "rating": r["initial_rating"],
                    "file_name": images[r["image_id"]]["file_name"],
                }
            )
        stats[len(chosen)] += 1

    with open(refs_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["user_id", "index", "image_id", "rating", "file_name"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"用户数: {len(rows) // max(1, max_refs)} | 每用户训练图数量分布: {dict(sorted(stats.items()))}")
    print(f"训练数据目录: {data_dir}")
    print(f"参考清单: {refs_path}（{len(rows)} 行）")


if __name__ == "__main__":
    main()
