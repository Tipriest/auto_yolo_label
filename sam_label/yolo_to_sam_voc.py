#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import shutil
import xml.etree.ElementTree as ET
from glob import glob
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from ultralytics import SAM

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None


def load_config(path: str) -> Dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping")
    return data


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path)


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def collect_images(image_dir: str, extensions: List[str]) -> List[str]:
    paths: List[str] = []
    for ext in extensions:
        pattern = os.path.join(image_dir, f"*.{ext}")
        paths.extend(glob(pattern))
    return sorted(set(paths))


def resolve_path(base: Optional[str], maybe_path: str) -> str:
    if os.path.isabs(maybe_path) or not base:
        return maybe_path
    return os.path.normpath(os.path.join(base, maybe_path))


def load_yolo_dataset_yaml(path: str) -> Tuple[str, str, Optional[str], List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("data.yaml must be a YAML mapping")
    base_path = data.get("path", "")
    train = data.get("train")
    val = data.get("val")
    names = data.get("names", [])
    if not train or not names:
        raise ValueError("data.yaml must include train and names")
    if not isinstance(names, list):
        raise ValueError("data.yaml names must be a list")
    train = resolve_path(base_path, train)
    val = resolve_path(base_path, val) if val else None
    return base_path, train, val, [str(n) for n in names]


def collect_split_images(source: str, extensions: List[str]) -> List[str]:
    if os.path.isfile(source):
        return [p for p in read_lines(source) if os.path.isfile(p)]
    if os.path.isdir(source):
        return collect_images(source, extensions)
    return []


def yolo_label_path(image_path: str, labels_root: Optional[str]) -> str:
    if labels_root:
        base = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(labels_root, f"{base}.txt")
    parts = image_path.split(os.sep)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        label_path = os.sep.join(parts)
        return os.path.splitext(label_path)[0] + ".txt"
    return os.path.splitext(image_path)[0] + ".txt"


def parse_yolo_labels(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    if not os.path.isfile(label_path):
        return []
    labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
            except ValueError:
                continue
            labels.append((cls_id, cx, cy, w, h))
    return labels


def yolo_to_xyxy(box: Tuple[float, float, float, float], width: int, height: int) -> List[float]:
    cx, cy, w, h = box
    x1 = (cx - w / 2.0) * width
    y1 = (cy - h / 2.0) * height
    x2 = (cx + w / 2.0) * width
    y2 = (cy + h / 2.0) * height
    return [max(0.0, x1), max(0.0, y1), min(float(width - 1), x2), min(float(height - 1), y2)]


def xyxy_to_voc_box(box: List[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(1, min(width, int(round(x1)) + 1))
    y1 = max(1, min(height, int(round(y1)) + 1))
    x2 = max(1, min(width, int(round(x2)) + 1))
    y2 = max(1, min(height, int(round(y2)) + 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def voc_palette() -> List[int]:
    palette = [0] * (256 * 3)
    for j in range(256):
        lab = j
        for i in range(8):
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            lab >>= 3
    return palette


def palette_colors() -> List[Tuple[int, int, int]]:
    palette = voc_palette()
    colors = []
    for i in range(256):
        r = palette[i * 3 + 0]
        g = palette[i * 3 + 1]
        b = palette[i * 3 + 2]
        colors.append((b, g, r))
    return colors


def save_mask(mask: "object", path: str) -> None:
    ensure_dir(os.path.dirname(path))
    if Image is not None:
        img = Image.fromarray(mask.astype("uint8"), mode="P")
        img.putpalette(voc_palette())
        img.save(path)
        return
    cv2.imwrite(path, mask)


def render_visualization(image: "object", mask: "object", alpha: float) -> "object":
    colors = palette_colors()
    color_mask = np.zeros_like(image)
    mask_ids = np.unique(mask)
    for class_id in mask_ids:
        if class_id == 0:
            continue
        color = colors[int(class_id) % len(colors)]
        color_mask[mask == class_id] = color
    blended = image.copy()
    fg = mask > 0
    blended[fg] = (image[fg] * (1 - alpha) + color_mask[fg] * alpha).astype("uint8")
    return blended


def write_voc_xml(
    image_path: str,
    base_name: str,
    width: int,
    height: int,
    depth: int,
    labels: List[Tuple[int, float, float, float, float]],
    names: List[str],
    output_path: str,
) -> None:
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "JPEGImages"
    ET.SubElement(annotation, "filename").text = os.path.basename(image_path)
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)
    ET.SubElement(annotation, "segmented").text = "1"

    for cls_id, cx, cy, w, h in labels:
        if cls_id < 0 or cls_id >= len(names):
            continue
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = str(names[cls_id])
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        ET.SubElement(obj, "occluded").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        x1, y1, x2, y2 = xyxy_to_voc_box(yolo_to_xyxy((cx, cy, w, h), width, height), width, height)
        ET.SubElement(bndbox, "xmin").text = str(x1)
        ET.SubElement(bndbox, "ymin").text = str(y1)
        ET.SubElement(bndbox, "xmax").text = str(x2)
        ET.SubElement(bndbox, "ymax").text = str(y2)

    tree = ET.ElementTree(annotation)
    ensure_dir(os.path.dirname(output_path))
    tree.write(output_path, encoding="utf-8", xml_declaration=False)


def build_coco_categories(names: List[str]) -> List[Dict]:
    categories = []
    for idx, name in enumerate(names, start=1):
        categories.append({"id": idx, "name": str(name), "supercategory": ""})
    return categories


def encode_coco_rle(binary: "object") -> Dict:
    h, w = binary.shape[:2]
    pixels = binary.reshape(-1, order="F").tolist()
    counts: List[int] = []
    count = 0
    prev = 0
    for pix in pixels:
        if pix != prev:
            counts.append(count)
            count = 1
            prev = pix
        else:
            count += 1
    counts.append(count)
    return {"size": [int(h), int(w)], "counts": counts}


def mask_to_coco_annotations(
    mask: "object",
    image_id: int,
    ann_id_start: int,
    segmentation_mode: str,
    is_crowd: int,
) -> Tuple[List[Dict], int]:
    annotations: List[Dict] = []
    ann_id = ann_id_start
    class_ids = np.unique(mask)
    for class_id in class_ids:
        if class_id == 0:
            continue
        binary = (mask == class_id).astype("uint8")
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if segmentation_mode == "rle" or is_crowd == 1:
            area = float(binary.sum())
            if area <= 0:
                continue
            ys, xs = np.where(binary > 0)
            if ys.size == 0:
                continue
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            rle = encode_coco_rle(binary)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(class_id),
                    "segmentation": rle,
                    "area": area,
                    "bbox": [float(x1), float(y1), float(x2 - x1 + 1), float(y2 - y1 + 1)],
                    "iscrowd": int(is_crowd),
                    "attributes": {},
                }
            )
            ann_id += 1
            continue
        for contour in contours:
            if contour.shape[0] < 3:
                continue
            area = float(cv2.contourArea(contour))
            if area <= 0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            segmentation = contour.reshape(-1, 2).flatten().astype(float).tolist()
            if len(segmentation) < 6:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(class_id),
                    "segmentation": [segmentation],
                    "area": area,
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "iscrowd": int(is_crowd),
                    "attributes": {},
                }
            )
            ann_id += 1
    return annotations, ann_id


def place_image(src: str, dst: str, copy_images: bool, use_symlinks: bool) -> None:
    if os.path.exists(dst):
        return
    ensure_dir(os.path.dirname(dst))
    if use_symlinks:
        os.symlink(src, dst)
    elif copy_images:
        shutil.copy2(src, dst)


def iter_pairs(items: Iterable[str]) -> Iterable[Tuple[str, str]]:
    for item in items:
        base = os.path.splitext(os.path.basename(item))[0]
        yield item, base


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="YOLO to SAM VOC segmentation dataset")
    parser.add_argument("-c", "--config", default="./yolo_to_sam_voc.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_cfg = cfg.get("data", {})
    sam_cfg = cfg.get("sam", {})
    out_cfg = cfg.get("output", {})

    dataset_yaml = data_cfg.get("dataset_yaml")
    if not dataset_yaml:
        raise ValueError("data.dataset_yaml is required")

    extensions = data_cfg.get("extensions", ["jpg", "jpeg", "png", "bmp"])
    labels_root = data_cfg.get("labels_root")
    skip_existing = bool(data_cfg.get("skip_existing", True))

    output_dir = out_cfg.get("output_dir")
    if not output_dir:
        raise ValueError("output.output_dir is required")

    copy_images = bool(out_cfg.get("copy_images", True))
    use_symlinks = bool(out_cfg.get("use_symlinks", False))
    save_visualizations = bool(out_cfg.get("save_visualizations", True))
    vis_dir_name = out_cfg.get("vis_dir", "Visualizations")
    vis_alpha = float(out_cfg.get("vis_alpha", 0.5))
    output_format = str(out_cfg.get("format", "voc")).strip().lower()
    if output_format not in {"voc", "coco"}:
        raise ValueError("output.format must be 'voc' or 'coco'")
    voc_save_masks = bool(out_cfg.get("voc_save_masks", True))
    coco_segmentation = str(out_cfg.get("coco_segmentation", "polygon")).strip().lower()
    if coco_segmentation not in {"polygon", "rle"}:
        raise ValueError("output.coco_segmentation must be 'polygon' or 'rle'")
    coco_is_crowd = int(out_cfg.get("coco_is_crowd", 0))
    coco_flatten_images = bool(out_cfg.get("coco_flatten_images", False))
    coco_file_name_only = bool(out_cfg.get("coco_file_name_only", False))

    base_path, train_src, val_src, names = load_yolo_dataset_yaml(dataset_yaml)

    if labels_root:
        labels_root = resolve_path(base_path, labels_root)

    train_images = collect_split_images(train_src, extensions)
    if not train_images:
        raise ValueError(f"No train images found in {train_src}")
    val_images = collect_split_images(val_src, extensions) if val_src else []

    if output_format == "voc":
        images_dir = os.path.join(output_dir, "JPEGImages")
        ann_dir = os.path.join(output_dir, "Annotations")
        sets_main_dir = os.path.join(output_dir, "ImageSets", "Main")
        sets_seg_dir = os.path.join(output_dir, "ImageSets", "Segmentation")
        masks_dir = os.path.join(output_dir, "SegmentationClass")
    else:
        images_dir = os.path.join(output_dir, "images")
        masks_dir = os.path.join(output_dir, "masks")
        ann_dir = None
        sets_main_dir = None
        sets_seg_dir = None
        annotations_dir = os.path.join(output_dir, "annotations")
    vis_dir = os.path.join(output_dir, vis_dir_name)
    vis_train_dir = os.path.join(vis_dir, "train")
    vis_val_dir = os.path.join(vis_dir, "val")

    ensure_dir(images_dir)
    if output_format == "voc":
        ensure_dir(ann_dir)
        ensure_dir(sets_main_dir)
        if voc_save_masks:
            ensure_dir(masks_dir)
            ensure_dir(sets_seg_dir)
    else:
        ensure_dir(masks_dir)
        ensure_dir(annotations_dir)
        if coco_flatten_images:
            ensure_dir(output_dir)
        else:
            ensure_dir(os.path.join(images_dir, "train"))
            ensure_dir(os.path.join(images_dir, "val"))
    if save_visualizations:
        ensure_dir(vis_train_dir)
        ensure_dir(vis_val_dir)

    weights = sam_cfg.get("weights", "sam2_b.pt")
    device = sam_cfg.get("device", "cuda")

    model = SAM(weights)
    model.to(device)

    def coco_image_paths(image_path: str, split_name: str) -> Tuple[str, str]:
        base_name = os.path.basename(image_path)
        if coco_flatten_images:
            return os.path.join(output_dir, base_name), base_name
        dst = os.path.join(images_dir, split_name, base_name)
        if coco_file_name_only:
            return dst, base_name
        return dst, os.path.join("images", split_name, base_name)

    def process_split(images: List[str], split_name: str) -> Optional[Dict]:
        split_list = []
        vis_root = vis_train_dir if split_name == "train" else vis_val_dir
        coco_images: List[Dict] = []
        coco_annotations: List[Dict] = []
        next_ann_id = 1
        next_image_id = 1
        for image_path, base in tqdm(list(iter_pairs(images)), desc=f"Processing {split_name}"):
            mask_path = os.path.join(masks_dir, f"{base}.png")
            if output_format == "voc":
                xml_path = os.path.join(ann_dir, f"{base}.xml")
                if skip_existing and os.path.isfile(xml_path):
                    split_list.append(base)
                    continue
            if output_format == "coco" and skip_existing and os.path.isfile(mask_path):
                split_list.append(base)
                continue

            image = cv2.imread(image_path)
            if image is None:
                continue
            height, width = image.shape[:2]
            label_path = yolo_label_path(image_path, labels_root)
            labels = parse_yolo_labels(label_path)

            if not labels:
                mask = (0 * image[:, :, 0]).astype("uint8")
                if output_format == "voc":
                    if voc_save_masks:
                        save_mask(mask, mask_path)
                    xml_path = os.path.join(ann_dir, f"{base}.xml")
                    write_voc_xml(image_path, base, width, height, image.shape[2], labels, names, xml_path)
                if output_format == "coco":
                    dst_image, file_name = coco_image_paths(image_path, split_name)
                else:
                    dst_image = os.path.join(images_dir, os.path.basename(image_path))
                place_image(image_path, dst_image, copy_images, use_symlinks)
                if save_visualizations:
                    vis_path = os.path.join(vis_root, f"{base}.jpg")
                    vis = render_visualization(image, mask, vis_alpha)
                    cv2.imwrite(vis_path, vis)
                if output_format == "coco":
                    coco_images.append(
                        {
                            "id": next_image_id,
                            "file_name": file_name,
                            "width": width,
                            "height": height,
                        }
                    )
                    next_image_id += 1
                split_list.append(base)
                continue

            boxes = [yolo_to_xyxy((cx, cy, w, h), width, height) for _, cx, cy, w, h in labels]

            results = model.predict(image, bboxes=boxes, verbose=False)
            masks = results[0].masks
            if masks is None:
                mask = (0 * image[:, :, 0]).astype("uint8")
            else:
                mask = (0 * image[:, :, 0]).astype("uint8")
                data = masks.data.cpu().numpy()
                for idx, (cls_id, _, _, _, _) in enumerate(labels):
                    if idx >= data.shape[0]:
                        break
                    class_value = int(cls_id) + 1
                    mask[data[idx] > 0.5] = class_value

            if output_format == "voc":
                if voc_save_masks:
                    save_mask(mask, mask_path)
                xml_path = os.path.join(ann_dir, f"{base}.xml")
                write_voc_xml(image_path, base, width, height, image.shape[2], labels, names, xml_path)
            if output_format == "coco":
                dst_image, file_name = coco_image_paths(image_path, split_name)
            else:
                dst_image = os.path.join(images_dir, os.path.basename(image_path))
            place_image(image_path, dst_image, copy_images, use_symlinks)
            if save_visualizations:
                vis_path = os.path.join(vis_root, f"{base}.jpg")
                vis = render_visualization(image, mask, vis_alpha)
                cv2.imwrite(vis_path, vis)
            if output_format == "coco":
                coco_images.append(
                    {
                        "id": next_image_id,
                        "file_name": file_name,
                        "width": width,
                        "height": height,
                    }
                )
                ann, next_ann_id = mask_to_coco_annotations(
                    mask,
                    next_image_id,
                    next_ann_id,
                    coco_segmentation,
                    coco_is_crowd,
                )
                coco_annotations.extend(ann)
                next_image_id += 1
            split_list.append(base)

        if output_format == "voc":
            return {"split_list": split_list, "split_name": split_name}
        return {
            "images": coco_images,
            "annotations": coco_annotations,
        }

    train_coco = process_split(train_images, "train")
    val_coco = process_split(val_images, "val") if val_images else None

    if output_format == "voc":
        all_names = ["background"] + [str(n) for n in names]
        with open(os.path.join(output_dir, "labelmap.txt"), "w", encoding="utf-8") as f:
            for name in all_names:
                f.write(f"{name}:::\n")
        default_list: List[str] = []
        train_list: List[str] = []
        val_list: List[str] = []
        if train_coco:
            train_list = train_coco.get("split_list", [])
            default_list.extend(train_list)
        if val_coco:
            val_list = val_coco.get("split_list", [])
            default_list.extend(val_list)
        default_path = os.path.join(sets_main_dir, "default.txt")
        with open(default_path, "w", encoding="utf-8") as f:
            f.write("\n".join(default_list))
        if voc_save_masks:
            train_path = os.path.join(sets_seg_dir, "train.txt")
            with open(train_path, "w", encoding="utf-8") as f:
                f.write("\n".join(train_list))
            if val_list:
                val_path = os.path.join(sets_seg_dir, "val.txt")
                with open(val_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(val_list))
        print(f"Done. VOC dataset saved to: {output_dir}")
    else:
        ensure_dir(os.path.join(images_dir, "train"))
        ensure_dir(os.path.join(images_dir, "val"))
        categories = build_coco_categories(names)
        coco_train = {
            "info": {},
            "licenses": [],
            "images": train_coco["images"] if train_coco else [],
            "annotations": train_coco["annotations"] if train_coco else [],
            "categories": categories,
        }
        with open(os.path.join(annotations_dir, "instances_train.json"), "w", encoding="utf-8") as f:
            json.dump(coco_train, f, ensure_ascii=True, indent=2)
        if val_coco is not None:
            coco_val = {
                "info": {},
                "licenses": [],
                "images": val_coco["images"],
                "annotations": val_coco["annotations"],
                "categories": categories,
            }
            with open(os.path.join(annotations_dir, "instances_val.json"), "w", encoding="utf-8") as f:
                json.dump(coco_val, f, ensure_ascii=True, indent=2)
        print(f"Done. COCO dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
