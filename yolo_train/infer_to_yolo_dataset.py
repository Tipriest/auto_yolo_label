import os
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Any

# 读取配置
try:
    import yaml
except Exception as e:
    print("缺少依赖：请先安装 pyyaml（pip install pyyaml）")
    raise e

try:
    from ultralytics import YOLO
except Exception as e:
    print("缺少依赖：请先安装 ultralytics（pip install ultralytics）")
    raise e


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"未找到配置文件：{config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # 默认值
    defaults = {
        "model": "",
        "source": "",
        "output": "",
        "splits": None,            # 列表或 null
        "imgsz": 640,
        "conf": 0.25,
        "iou": 0.7,
        "device": None,
        "batch": 16,
        "max_det": 300,
        "with_conf": False,
        "keep_empty": False
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    if not cfg["model"] or not cfg["source"] or not cfg["output"]:
        raise ValueError("配置文件中必须提供 model、source、output 字段")
    # 规范化 splits
    if isinstance(cfg.get("splits"), str):
        cfg["splits"] = [s.strip() for s in cfg["splits"].split(",") if s.strip()]
    return cfg


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def find_splits(images_root: Path) -> List[str]:
    if not images_root.exists():
        return []
    subs = [p.name for p in images_root.iterdir() if p.is_dir()]
    if subs:
        return subs
    imgs = [p for p in images_root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if imgs:
        return ['.']
    return []


def list_images(images_dir: Path) -> List[Path]:
    images = []
    if images_dir.is_dir():
        for p in images_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                images.append(p)
    elif images_dir.is_file() and images_dir.suffix.lower() in IMAGE_EXTS:
        images = [images_dir]
    return images


def save_yolo_labels(result, label_path: Path, add_conf: bool = False):
    lines = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "xywhn", None) is None or len(boxes) == 0:
        ensure_dir(label_path.parent)
        label_path.write_text("", encoding="utf-8")
        return
    xywhn = boxes.xywhn.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") and boxes.conf is not None else None
    for i, c in enumerate(clss):
        x, y, w, h = xywhn[i]
        if add_conf and confs is not None:
            lines.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {confs[i]:.6f}")
        else:
            lines.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    ensure_dir(label_path.parent)
    label_path.write_text("\n".join(lines), encoding="utf-8")


def copy_image(src: Path, dst: Path):
    ensure_dir(dst.parent)
    if not dst.exists():
        shutil.copy2(src, dst)


def copy_dataset_yaml(source_path: Path, output_root: Path, yaml_name: str = "dataset.yaml") -> None:
    candidates = []
    if source_path.is_file():
        candidates.append(source_path.parent / yaml_name)
    else:
        candidates.append(source_path / yaml_name)
        candidates.append(source_path.parent / yaml_name)

    dataset_yaml = next((p for p in candidates if p.exists()), None)
    if dataset_yaml is None:
        raise FileNotFoundError(f"未找到 {yaml_name}，请确认 source 目录内存在该文件")

    ensure_dir(output_root)
    shutil.copy2(dataset_yaml, output_root / yaml_name)


def main():
    # 确定配置文件路径：优先脚本同目录的 config.yaml，若未找到，尝试当前工作目录
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "infer_to_yolo_dataset.yaml"
    if not config_path.exists():
        alt = Path.cwd() / "infer_to_yolo_dataset.yaml"
        if alt.exists():
            config_path = alt
    print(f"读取配置文件：{config_path}")
    cfg = load_config(config_path)

    model_path = Path(cfg["model"])
    source_path = Path(cfg["source"])
    output_root = Path(cfg["output"])
    splits_cfg = cfg.get("splits")

    if not model_path.exists():
        print(f"模型不存在：{model_path}")
        sys.exit(1)

    # 解析源 images 根目录与 splits
    if source_path.is_file():
        images_root = source_path.parent
        splits = ['.']
    else:
        images_root = source_path / "images" if (source_path / "images").exists() else source_path
        if splits_cfg:
            splits = splits_cfg
        else:
            splits = find_splits(images_root)
            if not splits:
                print(f"未在 {images_root} 下发现可用的图像或 splits。")
                sys.exit(1)

    print(f"加载模型：{model_path}")
    model = YOLO(str(model_path))

    # 类别名
    names = {}
    if hasattr(model, 'names') and model.names:
        names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}

    processed_splits = []
    for sp in splits:
        if source_path.is_file():
            img_list = [source_path]
            in_split_dir = images_root
        else:
            in_split_dir = images_root if sp == '.' else (images_root / sp)
            if not in_split_dir.exists():
                print(f"跳过：未找到 images/{sp} -> {in_split_dir}")
                continue
            img_list = list_images(in_split_dir)
            if not img_list:
                print(f"跳过：{in_split_dir} 下未发现图像")
                continue

        out_split_name = 'predict' if sp == '.' else sp
        out_img_dir = output_root / "images" / out_split_name
        out_lbl_dir = output_root / "labels" / out_split_name
        ensure_dir(out_img_dir)
        ensure_dir(out_lbl_dir)

        print(f"开始推理 split='{sp}'，共 {len(img_list)} 张图像")
        results = model.predict(
            source=[str(p) for p in img_list],
            imgsz=int(cfg["imgsz"]),
            conf=float(cfg["conf"]),
            iou=float(cfg["iou"]),
            device=cfg["device"],
            max_det=int(cfg["max_det"]),
            save=False,
            stream=False,
            verbose=False,
            batch=int(cfg["batch"])
        )

        empty_count = 0
        for img_path, res in zip(img_list, results):
            out_img = out_img_dir / img_path.name
            out_lbl = out_lbl_dir / (img_path.stem + ".txt")

            if res.boxes is None or len(res.boxes) == 0:
                if bool(cfg["keep_empty"]):
                    ensure_dir(out_lbl.parent)
                    out_lbl.write_text("", encoding="utf-8")
                else:
                    empty_count += 1
            else:
                save_yolo_labels(res, out_lbl, add_conf=bool(cfg["with_conf"]))

            copy_image(img_path, out_img)

        kept = len(img_list) - empty_count if not bool(cfg["keep_empty"]) else len(img_list)
        print(f"完成 split='{sp}': 推理 {len(img_list)} 张，"
              f"{'丢弃无目标 ' + str(empty_count) + ' 张，' if not bool(cfg['keep_empty']) else ''}"
              f"保留 {kept} 张")

        processed_splits.append(sp)

    if not processed_splits:
        print("未处理任何 split，退出。")
        sys.exit(1)

    if not names:
        max_cls = -1
        for sp in processed_splits:
            out_split_name = 'predict' if sp == '.' else sp
            for txt in (output_root / "labels" / out_split_name).rglob("*.txt"):
                try:
                    for line in txt.read_text(encoding="utf-8").splitlines():
                        parts = line.strip().split()
                        if not parts:
                            continue
                        cid = int(float(parts[0]))
                        max_cls = max(max_cls, cid)
                except Exception:
                    pass
        names = {i: str(i) for i in range(max_cls + 1)} if max_cls >= 0 else {0: "0"}

    copy_dataset_yaml(source_path, output_root)
    print(f"已生成新的 YOLO 数据集：{output_root}")
    print(f"- images/ 与 labels/ 结构就绪")
    print(f"- dataset.yaml 已从 source 复制")


if __name__ == "__main__":
    main()