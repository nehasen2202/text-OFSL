import os
import json
from pathlib import Path

DATASETS = ["ND0", "ND1", "ND2", "ND3"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

ROOT = Path(__file__).resolve().parent


def read_classes(split_file):
    with open(split_file, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    if len(classes) != len(set(classes)):
        raise ValueError(f"Duplicate class names in {split_file}")

    return classes


def create_manifest(dataset_dir, classes, class_to_id):

    manifest = {
        "label_names": [],
        "image_names": [],
        "image_labels": []
    }

    for class_name in classes:

        class_dir = dataset_dir / "data" / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Class directory not found: {class_dir}"
            )

        images = sorted(
            path for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not images:
            raise ValueError(
                f"No images found in {class_dir}"
            )

        manifest["label_names"].append(class_name)

        for image_path in images:
            manifest["image_names"].append(
                str(image_path.resolve()).replace("\\", "/")
            )

            manifest["image_labels"].append(
                class_to_id[class_name]
            )

    return manifest


def save_json(data, output_path):

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def process_dataset(dataset_name):

    dataset_dir = ROOT / dataset_name
    split_dir = dataset_dir / "splits"

    print(f"\nProcessing {dataset_name}")

    train_classes = read_classes(split_dir / "train.txt")
    val_classes = read_classes(split_dir / "val.txt")
    test_classes = read_classes(split_dir / "test.txt")

    assert len(train_classes) == 3, "Expected 3 training classes"
    assert len(val_classes) == 2, "Expected 2 validation classes"
    assert len(test_classes) == 2, "Expected 2 testing classes"

    all_classes = (
        train_classes + val_classes + test_classes
    )

    if len(set(all_classes)) != 7:
        raise ValueError(
            f"{dataset_name}: Classes overlap between splits"
        )

    class_to_id = {
        class_name: idx
        for idx, class_name in enumerate(all_classes)
    }

    manifests = {
        "base.json": create_manifest(
            dataset_dir, train_classes, class_to_id
        ),
        "val.json": create_manifest(
            dataset_dir, val_classes, class_to_id
        ),
        "novel.json": create_manifest(
            dataset_dir, test_classes, class_to_id
        ),
        "all.json": create_manifest(
            dataset_dir, all_classes, class_to_id
        )
    }

    expected_images = 7000 if dataset_name == "ND3" else 4200

    if len(manifests["all.json"]["image_names"]) != expected_images:
        raise ValueError(
            f"{dataset_name}: Expected {expected_images} images, "
            f"found {len(manifests['all.json']['image_names'])}"
        )

    for filename, manifest in manifests.items():

        save_json(manifest, dataset_dir / filename)

        print(
            f"{filename}: "
            f"{len(manifest['label_names'])} classes, "
            f"{len(manifest['image_names'])} images"
        )

    print(f"{dataset_name} completed successfully")


if __name__ == "__main__":

    for dataset in DATASETS:
        process_dataset(dataset)

    print("\nAll dataset manifests generated successfully.")
