#!/usr/bin/env python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import argparse
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from transformers import BertTokenizer, BertModel

from utils.utils import *
from methods.conditionasimsiam import ConditionalSimSiam
from methods.IDEAL import IDEAL


# -----------------------------
# BERT TEXT ENCODER
# -----------------------------
tokenizer = None
bert_model = None


def init_bert(device):
    global tokenizer, bert_model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.eval()
    bert_model.to(device)


@torch.no_grad()
def vectorize_text(text, device):
    if text is None or str(text).strip() == "":
        return torch.zeros(768, device=device)

    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0)


# -----------------------------
# IMAGE HANDLING
# -----------------------------
def build_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def unnormalize_img(tensor):
    img = tensor.detach().cpu().clone()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy()


# -----------------------------
# DATA LOADING FROM JSON
# -----------------------------
def load_meta(json_file):
    with open(json_file, "r") as f:
        meta = json.load(f)

    image_names = meta["image_names"]
    image_labels = meta["image_labels"]
    image_data = meta.get("image_data", [""] * len(image_names))

    # Build original-label -> readable-class-name mapping.
    # Supported JSON fields:
    #   "class_names", "label_names", or "classes"
    # The field may be either a dictionary or a list.
    class_names = (
        meta.get("class_names")
        or meta.get("label_names")
        or meta.get("classes")
    )

    class_name_map = {}
    unique_labels = list(dict.fromkeys(image_labels))

    if isinstance(class_names, dict):
        for label, name in class_names.items():
            class_name_map[str(label)] = str(name)

    elif isinstance(class_names, list):
        # One class name per unique label, in first-occurrence order.
        if len(class_names) == len(unique_labels):
            for label, name in zip(unique_labels, class_names):
                class_name_map[str(label)] = str(name)

        # Class names indexed directly by integer label.
        elif all(
            isinstance(label, (int, np.integer))
            and 0 <= int(label) < len(class_names)
            for label in unique_labels
        ):
            for label in unique_labels:
                class_name_map[str(label)] = str(class_names[int(label)])

        # One class name supplied for every image.
        elif len(class_names) == len(image_labels):
            for label, name in zip(image_labels, class_names):
                class_name_map[str(label)] = str(name)

    by_class = defaultdict(list)

    for path, label, caption in zip(image_names, image_labels, image_data):
        label_key = str(label)

        # Fallback: use the image's parent-folder name.
        parent_folder = os.path.basename(os.path.dirname(path))
        class_name = class_name_map.get(
            label_key,
            parent_folder if parent_folder else label_key
        )

        by_class[label].append({
            "path": path,
            "label": label,
            "class_name": class_name,
            "caption": caption
        })

    return by_class


def make_episode(by_class, n_way, n_support, n_query, transform, device):
    available_classes = [c for c in by_class.keys() if len(by_class[c]) >= n_support + n_query]

    if len(available_classes) < n_way:
        raise ValueError(
            f"Not enough classes with at least {n_support + n_query} samples. "
            f"Found {len(available_classes)}, need {n_way}."
        )

    classes = random.sample(available_classes, n_way)

    x_episode = []
    y_episode = []
    text_episode = []
    meta_episode = []

    for epi_label, cls in enumerate(classes):
        samples = random.sample(by_class[cls], n_support + n_query)

        imgs = []
        texts = []
        metas = []

        for s in samples:
            img = Image.open(s["path"]).convert("RGB")
            img_t = transform(img)

            text_t = vectorize_text(s["caption"], device).detach().cpu()

            imgs.append(img_t)
            texts.append(text_t)
            metas.append({
                "path": s["path"],
                "caption": s["caption"],
                "true_original_label": s["label"],
                "true_class_name": s["class_name"],
                "episode_true_label": epi_label
            })

        x_episode.append(torch.stack(imgs))
        y_episode.append(torch.tensor([epi_label] * (n_support + n_query)))
        text_episode.append(torch.stack(texts))
        meta_episode.append(metas)

    x = torch.stack(x_episode).to(device)
    y = torch.stack(y_episode).to(device)
    text_vectors = torch.stack(text_episode).to(device)

    return x, y, text_vectors, classes, meta_episode


# -----------------------------
# MODEL LOADING
# -----------------------------
def load_ideal_model(args, image_size, model_name_internal, train_files, train_labels, allc_files, allc_labels):
    ssl_dir = (
        base_path
        + f"/save/checkpoints/{args.dataset}/{model_name_internal}_{args.pre_algorithm}_{args.ssl_algorithm}"
    )

    ssl_model = ConditionalSimSiam(
        model_func=model_dict[model_name_internal],
        n_way=args.train_n_way,
        n_support=args.n_shot,
        image_size=image_size,
        device=args.device
    )

    ssl_file = get_best_file(ssl_dir)
    assert ssl_file is not None, f"No SSL checkpoint found in {ssl_dir}"

    tmp = torch.load(ssl_file, map_location=args.device)
    ssl_model.load_state_dict(tmp["state"])
    ssl_model.eval()

    if args.noise_type == "OOD":
        meta_dir = (
            base_path
            + f"/save/checkpoints/{args.dataset}/"
              f"{model_name_internal}_{args.pre_algorithm}_{args.ssl_algorithm}_"
              f"{args.meta_algorithm}_OOT_{args.noises}_{args.attention_method}_{args.eta}_{args.gamma}"
        )
    else:
        meta_dir = (
            base_path
            + f"/save/checkpoints/{args.dataset}/"
              f"{model_name_internal}_{args.pre_algorithm}_{args.ssl_algorithm}_"
              f"{args.meta_algorithm}_{args.noise_type}_{args.noises}_{args.attention_method}_{args.eta}_{args.gamma}"
        )

    model = IDEAL(
        model_func=model_dict[model_name_internal],
        n_way=args.train_n_way,
        n_support=args.n_shot,
        image_size=image_size,
        outlier=args.noises,
        image_files=train_files,
        image_labels=train_labels,
        allc_files=allc_files,
        allc_labels=allc_labels,
        noise_type=args.noise_type,
        ssl_feature_extractor=ssl_model.feature_extractor,
        attention_method=args.attention_method,
        eta=args.eta,
        gamma=args.gamma,
        device=args.device
    )

    meta_file = get_best_file(meta_dir)
    assert meta_file is not None, f"No IDEAL checkpoint found in {meta_dir}"

    tmp = torch.load(meta_file, map_location=args.device)
    model.load_state_dict(tmp["state"])
    model.eval()

    print(f"[LOADED SSL]   {ssl_file}")
    print(f"[LOADED IDEAL] {meta_file}")

    return model


# -----------------------------
# COLLECT QUALITATIVE CASES
# -----------------------------
@torch.no_grad()
def collect_cases(model, by_class, args, image_size):
    transform = build_transform(image_size)

    correct_cases = []
    wrong_cases = []

    for ep in range(args.num_episodes):
        x, y, text_vectors, classes, meta_episode = make_episode(
            by_class=by_class,
            n_way=args.test_n_way,
            n_support=args.n_shot,
            n_query=args.n_query,
            transform=transform,
            device=args.device
        )

        model.text_vectors = text_vectors
        model.n_query = x.size(1) - args.n_shot
        model.n_way = x.size(0)

        scores, scores_ssl, scores_sl, textual_scores = model.set_forward(x)

        probs = F.softmax(scores, dim=1)
        confs, preds = probs.max(dim=1)

        y_query = np.repeat(range(model.n_way), model.n_query)

        query_metas = []
        query_imgs = []

        for cls_idx in range(model.n_way):
            for q_idx in range(args.n_shot, args.n_shot + model.n_query):
                query_metas.append(meta_episode[cls_idx][q_idx])
                query_imgs.append(x[cls_idx, q_idx].detach().cpu())

        for i in range(len(y_query)):
            pred_epi = int(preds[i].cpu().item())
            true_epi = int(y_query[i])
            conf = float(confs[i].cpu().item())

            true_class_value = classes[true_epi]
            pred_class_value = classes[pred_epi]

            # All samples belonging to an episodic class share the same
            # original label and readable class name.
            true_class_name = meta_episode[true_epi][0]["true_class_name"]
            pred_class_name = meta_episode[pred_epi][0]["true_class_name"]

            case = {
                "image_tensor": query_imgs[i],
                "image_path": query_metas[i]["path"],
                "caption": query_metas[i]["caption"],

                # Temporary labels used only inside the current episode.
                "true_episode_label": true_epi,
                "pred_episode_label": pred_epi,

                # Original dataset class values.
                "true_class_value": true_class_value,
                "pred_class_value": pred_class_value,

                # Human-readable class names.
                "true_class_name": true_class_name,
                "pred_class_name": pred_class_name,

                "confidence": conf,
                "correct": pred_epi == true_epi
            }

            # Collect diverse cases: maximum one correct and one wrong per true class
            true_class_name = case["true_class_name"]

            already_correct_classes = {
                c["true_class_name"] for c in correct_cases
            }

            already_wrong_classes = {
                c["true_class_name"] for c in wrong_cases
            }

            if pred_epi == true_epi:
                if (
                    true_class_name not in already_correct_classes
                    and len(correct_cases) < args.num_correct
                ):
                    correct_cases.append(case)
            else:
                if (
                    true_class_name not in already_wrong_classes
                    and len(wrong_cases) < args.num_wrong
                ):
                    wrong_cases.append(case)

        print(
            f"Episode {ep + 1}/{args.num_episodes} | "
            f"correct collected: {len(correct_cases)} | "
            f"wrong collected: {len(wrong_cases)}"
        )

        print("Correct classes:", [c["true_class_name"] for c in correct_cases])
        print("Wrong classes:", [c["true_class_name"] for c in wrong_cases])

        if len(correct_cases) >= args.num_correct and len(wrong_cases) >= args.num_wrong:
            break

    return correct_cases, wrong_cases


# -----------------------------
# PAPER-READY FIGURE
# -----------------------------
def wrap_text(text, width=58):
    text = str(text)
    words = text.split()
    lines = []
    current = []

    for w in words:
        if len(" ".join(current + [w])) <= width:
            current.append(w)
        else:
            lines.append(" ".join(current))
            current = [w]

    if current:
        lines.append(" ".join(current))

    return "\n".join(lines)


def plot_cases_paper_ready(correct_cases, wrong_cases, out_path):
    cases = wrong_cases + correct_cases

    if len(cases) == 0:
        raise ValueError("No cases collected. Increase --num_episodes.")

    n = len(cases)
    cols = 2
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(14, rows * 7))

    for i, c in enumerate(cases):
        ax = plt.subplot(rows, cols, i + 1)

        img = Image.open(c["image_path"]).convert("RGB")
        ax.imshow(img)
        ax.axis("off")

        status = "Correct" if c["correct"] else "Misclassified"
        caption = wrap_text(c["caption"], width=60)

        title = (
            f"{status}\n"
            f"True: {c['true_class_name']} "
            f"(value: {c['true_class_value']})\n"
            f"Predicted: {c['pred_class_name']} "
            f"(value: {c['pred_class_value']})\n"
            f"Confidence: {c['confidence']:.2f}\n"
            f"Caption: {caption}"
        )

        ax.set_title(title, fontsize=10, pad=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    print(f"[SAVED PAPER-READY FIGURE] {out_path}")

def save_individual_cases_bottom_text(correct_cases, wrong_cases, output_dir):
    """
    Saves each selected case as a separate high-resolution PNG.
    Layout:
        IMAGE
        analysis text at bottom
    """

    os.makedirs(output_dir, exist_ok=True)

    cases = wrong_cases + correct_cases

    for idx, c in enumerate(cases):
        status = "CORRECT" if c["correct"] else "MISCLASSIFIED"

        img = Image.open(c["image_path"]).convert("RGB")

        caption = wrap_text(c["caption"], width=75)

        analysis_text = (
            f"Status: {status}\n"
            f"True Class: {c['true_class_name']} "
            f"(value: {c['true_class_value']})\n"
            f"Predicted Class: {c['pred_class_name']} "
            f"(value: {c['pred_class_value']})\n"
            f"Confidence Score: {c['confidence']:.4f}\n"
            f"Generated Caption: {caption}"
        )

        fig = plt.figure(figsize=(10, 11))

        ax_img = plt.subplot2grid((5, 1), (0, 0), rowspan=4)
        ax_img.imshow(img)
        ax_img.axis("off")

        ax_text = plt.subplot2grid((5, 1), (4, 0), rowspan=1)
        ax_text.axis("off")

        ax_text.text(
            0.01,
            0.98,
            analysis_text,
            fontsize=22,
            fontweight="bold",
            color="black",
            va="top",
            ha="left",
            wrap=True
        )

        save_path = os.path.join(
            output_dir,
            f"case_{idx + 1}_{status.lower()}.png"
        )

        plt.savefig(save_path, dpi=600, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"[SAVED INDIVIDUAL CASE] {save_path}")


# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="tieredImagenet",
                        choices=["cifar", "fc100", "miniImagenet", "tieredImagenet"])
    parser.add_argument("--noises", type=int, default=1)
    parser.add_argument("--noise_type", type=str, default="IT",
                        choices=["IT", "OOT", "OOD"])
    parser.add_argument("--model_name", type=str, default="Conv4",
                        choices=["Conv4", "ResNet12"])
    parser.add_argument("--train_n_way", type=int, default=5)
    parser.add_argument("--test_n_way", type=int, default=5)
    parser.add_argument("--n_shot", type=int, default=5)
    parser.add_argument("--n_query", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pre_algorithm", type=str, default="baseline++")
    parser.add_argument("--ssl_algorithm", type=str, default="conditionalsimsiam")
    parser.add_argument("--meta_algorithm", type=str, default="IDEAL")
    parser.add_argument("--attention_method", type=str, default="bilstm",
                        choices=["bilstm", "transformer"])
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.1)

    parser.add_argument("--split", type=str, default="novel",
                        choices=["base", "val", "novel", "allc"])
    parser.add_argument("--num_episodes", type=int, default=30)
    parser.add_argument("--num_wrong", type=int, default=4)
    parser.add_argument("--num_correct", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="error_analysis_outputs")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    if not torch.cuda.is_available():
        print("No GPU detected. Switching to CPU.")
        args.device = "cpu"
    else:
        print("GPU detected.")

    init_bert(args.device)

    image_size = get_image_size(model_name=args.model_name, dataset=args.dataset)
    model_name_internal = get_model_name(model_name=args.model_name, dataset=args.dataset)

    base_file, val_file = get_train_files(dataset=args.dataset)

    train_meta = json.load(open(base_file))
    train_files = train_meta["image_names"]
    train_labels = train_meta["image_labels"]

    eval_file = get_novel_file(dataset=args.dataset, split=args.split)

    eval_meta = json.load(open(eval_file))
    eval_files = eval_meta["image_names"]
    eval_labels = eval_meta["image_labels"]

    if "image_data" not in eval_meta:
        print("[WARNING] image_data not found in JSON. Captions will be empty.")

    allc_files = eval_files
    allc_labels = eval_labels

    model = load_ideal_model(
        args=args,
        image_size=image_size,
        model_name_internal=model_name_internal,
        train_files=train_files,
        train_labels=train_labels,
        allc_files=allc_files,
        allc_labels=allc_labels
    )

    by_class = load_meta(eval_file)

    correct_cases, wrong_cases = collect_cases(
        model=model,
        by_class=by_class,
        args=args,
        image_size=image_size
    )

    fig_path = os.path.join(args.output_dir, "error_analysis_figure.png")

    plot_cases_paper_ready(
        correct_cases=correct_cases,
        wrong_cases=wrong_cases,
        out_path=fig_path
    )
    
    save_individual_cases_bottom_text(
    correct_cases=correct_cases,
    wrong_cases=wrong_cases,
    output_dir=args.output_dir
)

    print("\nDone.")
    print(f"Figure saved at: {fig_path}")


if __name__ == "__main__":
    main()
