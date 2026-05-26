"""
Push trained Multi-Head-CRF models to HuggingFace.

Structure:
    8 repos (one per language + MIXED), each with models as branches.
    Best model → main branch. All others → named branches.
    All repos grouped in a HuggingFace Collection.

Usage:
    python push_models.py --org YOUR_HF_ORG [--dry-run]
    python push_models.py --org YOUR_HF_ORG --collection-only  # just create the collection

Assumes you are logged in via `huggingface-cli login`.
"""

import argparse
import os
import glob
from huggingface_hub import HfApi, create_repo, create_collection

MODELS_BASE = os.path.join(os.path.dirname(__file__), "trained-models")

# Best performing model per language (by strict-match F1 on validation set).
BEST_LANG_RUN = {
    "ES": ("lcampillos-C64-H3-E30-Aukn-%0.25-P0.2-42", 0.8071),
    "EN": ("microsoft-BiomedNLP-PubMedBERT-large-uncased-abstract-C64-H3-E30-Arandom-%0.25-P0.5-42", 0.7368),
    "CZ": ("ufal-C64-H3-E60-Arandom-%0.1-P0.5-42", 0.6998),
    "IT": ("IVN-RIN-C64-H1-E60-Arandom-%0.1-P0.2-42", 0.7062),
    "SV": ("KB-C64-H3-E60-Arandom-%0.1-P0.5-42", 0.7051),
    "NL": ("CLTL-C64-H3-E60-Arandom-%0.25-P0.2-42", 0.6898),
    "RO": ("dumitrescustefan-C64-H3-E60-Aukn-%0.1-P0.2-42", 0.6758),
}
BEST_MULTILINGUAL_RUN = ("FacebookAI-xlm-roberta-large-C64-H3-E3-Arandom-%0.25-P0.2-42", None)

# Which models to include per language, matching the inference scripts.
LANG_FILTERS = {
    "CZ": {},  # all 20
    "EN": {"include_substr": ["large"]},  # only "large" models → 20
    "ES": {"include_substr": ["E30"]},  # only E30 → 20
    "IT": {
        "exclude_exact": [
            "IVN-RIN-C64-H1-E60-Arandom-%0.25-P0.5-999",
            "IVN-RIN-C64-H1-E60-Aukn-%0.1-P0.5-999",
        ]
    },  # 14
    "NL": {
        "exclude_exact": [
            "CLTL-C64-H3-E60-Arandom-%0.25-P0.2-456",
            "CLTL-C64-H3-E60-Arandom-%0.1-P0.5-999",
        ]
    },  # 18
    "RO": {
        "exclude_exact": [
            "dumitrescustefan-C64-H3-E60-Arandom-%0.25-P0.2-999",
            "dumitrescustefan-C64-H3-E60-Arandom-%0.25-P0.5-999",
        ],
        "exclude_substr": ["KB", "kb"],
    },  # 18
    "SV": {
        "exclude_exact": [
            "KB-C64-H3-E60-Aukn-%0.25-P0.2-456",
            "KB-C64-H3-E60-Arandom-%0.1-P0.5-456",
            "KB-C64-H3-E60-Aukn-%0.1-P0.5-456",
            "KB-C64-H3-E60-Arandom-%0.1-P0.5-999",
            "KB-C64-H3-E60-Arandom-%0.25-P0.2-123",
            "KB-C64-H3-E60-Arandom-%0.25-P0.2-456",
            "KB-C64-H3-E60-Arandom-%0.1-P0.2-456",
            "KB-C64-H3-E60-Aukn-%0.1-P0.5-999",
            "KB-C64-H3-E60-Aukn-%0.25-P0.2-999",
        ]
    },  # 11
    "MIXED": {
        "include_substr": ["large", "E3"],  # must match ALL → large E3 models
        "include_mode": "all",
        "exclude_exact": [
            "FacebookAI-xlm-roberta-large-C64-H3-E3-Aukn-%0.1-P0.2-123",
        ],
    },  # 7
}

IGNORE_PATTERNS = ["rng_state.pth", "scheduler.pt", "training_args.bin", "trainer_state.json"]


def sanitize_branch(name: str) -> str:
    """Replace characters invalid in HF revision names."""
    return name.replace("%", "pct")


def should_include(model_name: str, filters: dict) -> bool:
    if not filters:
        return True

    if "include_substr" in filters:
        mode = filters.get("include_mode", "all")
        matches = [s in model_name for s in filters["include_substr"]]
        if mode == "all" and not all(matches):
            return False
        elif mode != "all" and not any(matches):
            return False

    if model_name in filters.get("exclude_exact", []):
        return False

    for sub in filters.get("exclude_substr", []):
        if sub in model_name:
            return False

    return True


def get_latest_checkpoint(model_dir: str) -> str | None:
    checkpoints = sorted(
        glob.glob(os.path.join(model_dir, "checkpoint-*")),
        key=lambda p: int(os.path.basename(p).split("-")[-1]),
    )
    if checkpoints:
        return checkpoints[-1]
    if os.path.isfile(os.path.join(model_dir, "config.json")):
        return model_dir
    return None


def get_best_name(lang: str) -> str | None:
    if lang == "MIXED":
        return BEST_MULTILINGUAL_RUN[0]
    entry = BEST_LANG_RUN.get(lang)
    return entry[0] if entry else None


def get_best_f1(lang: str) -> float | None:
    if lang == "MIXED":
        return BEST_MULTILINGUAL_RUN[1]
    entry = BEST_LANG_RUN.get(lang)
    return entry[1] if entry else None


def collect_models_by_lang() -> dict[str, list[dict]]:
    by_lang = {}
    for lang, filters in LANG_FILTERS.items():
        full_dir = os.path.join(MODELS_BASE, lang, "full")
        if not os.path.isdir(full_dir):
            print(f"WARNING: {full_dir} not found, skipping {lang}")
            continue

        models = []
        for entry in sorted(os.listdir(full_dir)):
            entry_path = os.path.join(full_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            if not should_include(entry, filters):
                continue

            ckpt = get_latest_checkpoint(entry_path)
            if ckpt is None or not os.path.isfile(os.path.join(ckpt, "config.json")):
                print(f"WARNING: No valid checkpoint for {lang}/{entry}, skipping")
                continue

            models.append({"name": entry, "checkpoint_path": ckpt})

        # Sort so best model is first
        best_name = get_best_name(lang)
        models.sort(key=lambda m: (0 if m["name"] == best_name else 1, m["name"]))
        by_lang[lang] = models

    return by_lang


def make_model_card(org: str, lang: str, models: list[dict]) -> str:
    best_name = get_best_name(lang)
    best_f1 = get_best_f1(lang)

    if lang == "MIXED":
        title = "MultiClinNER Multilingual Models"
        description = "Multilingual clinical NER models trained on all 7 languages (CZ, EN, ES, IT, NL, RO, SV)."
    else:
        title = f"MultiClinNER {lang} Models"
        description = f"Clinical NER models for {lang}, trained with Multi-Head CRF architecture."

    f1_line = f"- **Best F1**: {best_f1:.4f}\n" if best_f1 is not None else ""

    branch_table = "| Branch | Model | Best? |\n|--------|-------|-------|\n"
    for m in models:
        is_best = m["name"] == best_name
        branch = "main" if is_best else sanitize_branch(m["name"])
        star = "**Yes**" if is_best else ""
        branch_table += f"| `{branch}` | `{m['name']}` | {star} |\n"

    return (
        f"---\n"
        f"tags:\n"
        f"  - named-entity-recognition\n"
        f"  - clinical-nlp\n"
        f"  - multiclinner\n"
        f"  - multi-head-crf\n"
        f"  - token-classification\n"
        f"language:\n"
        f"  - {'multilingual' if lang == 'MIXED' else lang.lower()}\n"
        f"license: apache-2.0\n"
        f"---\n\n"
        f"# {title}\n\n"
        f"{description}\n\n"
        f"## Best Model\n\n"
        f"- **Model**: `{best_name}`\n"
        f"{f1_line}"
        f"- **Branch**: `main`\n\n"
        f"## Usage\n\n"
        f"```python\n"
        f"# Load the best model (main branch)\n"
        f'from transformers import AutoTokenizer, AutoModelForTokenClassification\n\n'
        f'model = AutoModelForTokenClassification.from_pretrained("{org}/MultiClinNER-{lang}")\n'
        f'tokenizer = AutoTokenizer.from_pretrained("{org}/MultiClinNER-{lang}")\n\n'
        f"# Load a specific model variant\n"
        f'model = AutoModelForTokenClassification.from_pretrained("{org}/MultiClinNER-{lang}", revision="BRANCH_NAME")\n'
        f"```\n\n"
        f"## All Models ({len(models)} variants)\n\n"
        f"{branch_table}"
    )


def push_models(org: str, dry_run: bool = False):
    api = HfApi()
    by_lang = collect_models_by_lang()

    total = sum(len(ms) for ms in by_lang.values())
    print(f"Found {total} models across {len(by_lang)} repos:\n")
    for lang, models in by_lang.items():
        best = get_best_name(lang)
        print(f"  MultiClinNER-{lang}: {len(models)} branches (best: {best})")
    print()

    if dry_run:
        print("--- DRY RUN ---\n")
        for lang, models in by_lang.items():
            repo_id = f"{org}/MultiClinNER-{lang}"
            print(f"  Repo: {repo_id}")
            best_name = get_best_name(lang)
            for m in models:
                is_best = m["name"] == best_name
                branch = "main" if is_best else sanitize_branch(m["name"])
                f1 = get_best_f1(lang) if is_best else None
                star = f" <- BEST (F1={f1:.4f})" if is_best and f1 else (" <- BEST" if is_best else "")
                print(f"    branch: {branch}{star}")
                print(f"      <- {m['checkpoint_path']}")
            print()
        print(f"Total: {len(by_lang)} repos, {total} branches")
        return

    pushed_repos = []
    counter = 0

    for lang, models in by_lang.items():
        repo_id = f"{org}/MultiClinNER-{lang}"
        best_name = get_best_name(lang)
        print(f"\n{'='*60}")
        print(f"REPO: {repo_id} ({len(models)} models)")
        print(f"{'='*60}")

        create_repo(repo_id, repo_type="model", exist_ok=True)

        for m in models:
            counter += 1
            is_best = m["name"] == best_name
            branch = "main" if is_best else sanitize_branch(m["name"])
            tag = " [BEST -> main]" if is_best else f" [branch: {branch}]"
            print(f"  [{counter}/{total}] {m['name']}{tag}")

            if is_best:
                # Best model goes to main branch
                api.upload_folder(
                    folder_path=m["checkpoint_path"],
                    repo_id=repo_id,
                    repo_type="model",
                    ignore_patterns=IGNORE_PATTERNS,
                )
                # Upload model card to main
                card = make_model_card(org, lang, models)
                api.upload_file(
                    path_or_fileobj=card.encode(),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="model",
                )
            else:
                # Create the branch from main, then upload
                try:
                    api.create_branch(repo_id, repo_type="model", branch=branch)
                except Exception:
                    pass  # branch may already exist
                api.upload_folder(
                    folder_path=m["checkpoint_path"],
                    repo_id=repo_id,
                    repo_type="model",
                    revision=branch,
                    create_pr=False,
                    ignore_patterns=IGNORE_PATTERNS,
                )

            print(f"    Done.")

        pushed_repos.append(repo_id)

    # Create collection
    print(f"\n{'='*60}")
    print("Creating HuggingFace Collection...")
    print(f"{'='*60}")

    collection = create_collection(
        title="MultiClinNER Models",
        namespace=org,
        description=(
            "Multi-Head CRF models for clinical Named Entity Recognition "
            "across 7 languages (CZ, EN, ES, IT, NL, RO, SV) plus multilingual. "
            "Each repo contains the best model on main and all variants as branches."
        ),
        exists_ok=True,
    )

    for repo_id in pushed_repos:
        try:
            api.add_collection_item(
                collection.slug,
                item_id=repo_id,
                item_type="model",
                exists_ok=True,
            )
        except Exception as e:
            print(f"  Warning: could not add {repo_id} to collection: {e}")

    print(f"\nCollection: https://huggingface.co/collections/{collection.slug}")
    print(f"\nAll {total} models pushed across {len(pushed_repos)} repos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push Multi-Head-CRF models to HuggingFace")
    parser.add_argument("--org", required=True, help="HuggingFace organization name")
    parser.add_argument("--dry-run", action="store_true", help="List repos/branches without pushing")
    args = parser.parse_args()
    push_models(args.org, args.dry_run)
