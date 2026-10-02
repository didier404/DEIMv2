"""Generate a small smoke dataset from crn_valley_1280_coco (hard-links, no copies)."""

import json
from pathlib import Path

SOURCE = Path("F:/Data/Small_fetting_defect/patch/crn_valley_1280_coco")
TARGET = Path("F:/Data/Small_fetting_defect/patch/smoke_valley_1280_coco")
SIZES = {"train": 64, "valid": 32}

for split, count in SIZES.items():
    src = SOURCE / split
    dst = TARGET / split
    (dst / "images").mkdir(parents=True, exist_ok=True)
    annotation_path = src / "_annotations.coco.json"
    ann = json.loads(annotation_path.read_text(encoding="utf-8"))
    keep_images = ann["images"][:count]
    keep_ids = {img["id"] for img in keep_images}
    ann["images"] = keep_images
    ann["annotations"] = [a for a in ann["annotations"] if a["image_id"] in keep_ids]
    for img in keep_images:
        link = dst / img["file_name"]
        if not link.exists():
            link.hardlink_to(src / img["file_name"])
    (dst / "_annotations.coco.json").write_text(
        json.dumps(ann, ensure_ascii=False), encoding="utf-8"
    )
    print(f"{split}: {len(keep_images)} images, {len(ann['annotations'])} annotations")
