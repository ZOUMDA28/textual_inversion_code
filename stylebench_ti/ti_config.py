"""Textual Inversion (SDXL) 复现配置。

说明：官方仓库 rinongal/textual_inversion 只支持 LDM/SD1.x（README 明确 SD 支持为 WIP），
无法直接训练 SDXL。这里沿用其方法（学习一个文本 embedding），训练实现采用
diffusers 官方 SDXL 版 textual inversion 脚本（ti_trainer/textual_inversion_sdxl.py）。
"""

# ---------------- 数据与计划（复用 IP-Adapter/InstantStyle 的测试集与内容选择） ----------------
STYLEBENCH_DIR = "/data-nfs/gpu4-0/ui202420199/stylebench"
PLAN_FILE = f"{STYLEBENCH_DIR}/manifest/generation_plan.jsonl"   # 58 个 (user, content)

# ---------------- 模型 ----------------
MODELS_ROOT = "/data-nfs/gpu4-0/ui202420199/models"
SDXL_CKPT = f"{MODELS_ROOT}/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors"
SDXL_DIFFUSERS = f"{MODELS_ROOT}/stable-diffusion-xl-base-1.0-diffusers"  # 由 convert_sdxl.py 生成

TRAINER_SCRIPT = "/data-nfs/gpu4-0/ui202420199/ti_trainer/textual_inversion_sdxl.py"

# ---------------- 训练数据选择 ----------------
TI_SELECT_SEED = 2026          # 固定随机种子：4–5 分图像超过 5 张时用于选 5 张
ALLOWED_RATINGS = (4, 5)       # stage1 评分 4–5
MAX_REFS = 5                   # 最多 5 张

# ---------------- Textual Inversion 训练 ----------------
INITIALIZER_TOKEN = "style"    # 初始化词（原论文建议用一个粗略描述词）
LEARNABLE_PROPERTY = "style"   # 学习"风格"
TRAIN_RESOLUTION = 1024
TRAIN_STEPS = 2000             # 原论文 ~5000 步（其他数据集）；SDXL+风格训练社区主流 1500-3000 步，本任务取 2000
TRAIN_LR = 5e-4                # diffusers 官方 TI 训练默认值；此前误用 LoRA 的 1e-4，导致风格信号过弱
TRAIN_BATCH_SIZE = 1
TRAIN_SEED = 42

# ---------------- 生成协议（与 IP-Adapter / InstantStyle 保持一致） ----------------
WIDTH = 1024
HEIGHT = 1024
NUM_STEPS = 30
CFG = 5.0
SAMPLER = "EulerDiscreteScheduler"
NEGATIVE_PROMPT = (
    "text, watermark, lowres, low quality, worst quality, deformed, glitch, "
    "low contrast, noisy, saturation, blurry"
)
GEN_SEED = 42
PROMPT_SUFFIX = "in the style of {token}"   # 内容描述 + 学到的风格 token

# ---------------- 输出目录 ----------------
BASE_DIR = "/data-nfs/gpu4-0/ui202420199/stylebench_ti"
DATA_DIR = f"{BASE_DIR}/ti_data"            # 每用户训练图（软链接）
EMBED_DIR = f"{BASE_DIR}/ti_embeddings"     # 每用户学到的 embedding
OUTPUT_ROOT = "/data-nfs/gpu4-0/ui202420199/Textual Inversion-result"
RESULTS_JSONL = f"{OUTPUT_ROOT}/results.jsonl"

# ---------------- 3 参考图 smoke test（独立目录，不影响上面的 5 图正式结果） ----------------
SMOKE_MAX_REFS = 3             # 每用户 3 张
SMOKE_SELECT_MODE = "rating5_reverse"   # 只取 rating==5，按 stage1 列表位置从后往前
SMOKE_DATA_DIR = f"{BASE_DIR}/ti_data_3ref"
SMOKE_EMBED_DIR = f"{BASE_DIR}/ti_embeddings_3ref"
SMOKE_OUTPUT_ROOT = f"{OUTPUT_ROOT}/_smoke3ref"
SMOKE_RESULTS_JSONL = f"{SMOKE_OUTPUT_ROOT}/results.jsonl"
SMOKE_PLAN = f"{BASE_DIR}/smoke3ref_plan.jsonl"


def placeholder_token(user_id: str) -> str:
    """每个用户一个独立的占位词，例如 <u0015>。"""
    return f"<{user_id.lower()}>"


def protocol_dict(max_refs: int = None, select_mode: str = None) -> dict:
    return {
        "algorithm": "textual_inversion",
        "backbone": "SDXL",
        "sdxl_ckpt": SDXL_CKPT,
        "learnable_property": LEARNABLE_PROPERTY,
        "initializer_token": INITIALIZER_TOKEN,
        "train_steps": TRAIN_STEPS,
        "train_lr": TRAIN_LR,
        "train_resolution": TRAIN_RESOLUTION,
        "train_seed": TRAIN_SEED,
        "select_seed": TI_SELECT_SEED,
        "select_mode": select_mode or "rating45_hash",
        "max_refs": max_refs if max_refs is not None else MAX_REFS,
        "allowed_ratings": list(ALLOWED_RATINGS),
        "gen_width": WIDTH,
        "gen_height": HEIGHT,
        "gen_steps": NUM_STEPS,
        "gen_cfg": CFG,
        "gen_sampler": SAMPLER,
        "gen_seed": GEN_SEED,
        "prompt": "content_groups.json:prompt.en + ' in the style of <token>'",
    }
