"""
prepare_submissions.py
======================
Prepares 5 submissions per language per entity type for MultiClinNER.

All wandb best-run selections are HARDCODED from the uploaded wandb exports.

Submission types:
  1. Best multilingual model  (FacebookAI-xlm-roberta-large, fixed config)
  2. Best language-specific model  (highest val F1 per language from wandb)
  3. Ensemble of all multilingual (non-roberta) runs  (entity-level)
  4. Ensemble of all roberta runs  (entity-level)
  5. Ensemble of ALL runs in the directory  (entity-level)

Usage:
  python prepare_submissions.py \
      --predictions_root ~/MultiClinNer/Multi-Head-CRF/src/predictions_test \
      --dataset_root /home/id.aau.dk/kr75cs/MultiClinNer/Multi-Head-CRF/dataset/MultiClinAI-training+NER_test_bg_v1.2_260318/MultiClinNER \
      --output_root ./submissions

Optional flags:
  --lang es          process only one language
  --entity disease   process only one entity type
"""

import os, re, glob, json, csv
from collections import defaultdict
import click
from tqdm import tqdm
import concurrent.futures
from ensemble import ensemble_entity_level

# ---------------------------------------------------------------------------
# HARDCODED CONFIG (derived from uploaded wandb CSVs)
# ---------------------------------------------------------------------------

LANGUAGES = ["es", "en", "cz", "it", "nl", "ro", "sv"]
ENTITIES  = ["disease", "procedure", "symptom"]

# Prediction subfolder names on disk
LANG_FOLDER = {
    "es": "ES", "en": "EN", "cz": "CZ",
    "it": "IT", "nl": "NL", "ro": "RO", "sv": "SV",
}

# Fixed best multilingual model (no wandb run — use this exact config)
MULTILINGUAL_MODEL    = "FacebookAI-xlm-roberta-large"
BEST_MULTILINGUAL_RUN = "FacebookAI-xlm-roberta-large-C64-H3-E3-Arandom-%0.25-P0.2-42"

# Best language-specific run per language (by highest val F1 from wandb exports)
BEST_LANG_RUN = {
    "es": "lcampillos-C64-H3-E30-Aukn-%0.25-P0.2-42",         # F1=0.8071
    "en": "microsoft-BiomedNLP-PubMedBERT-large-uncased-abstract-C64-H3-E30-Arandom-%0.25-P0.5-42",          # F1=0.7368
    "cz": "ufal-C64-H3-E60-Arandom-%0.1-P0.5-42",             # F1=0.6998
    "it": "IVN-RIN-C64-H1-E60-Arandom-%0.1-P0.2-42",          # F1=0.7062
    "sv": "KB-C64-H3-E60-Arandom-%0.1-P0.5-42",               # F1=0.7051
    "nl": "CLTL-C64-H3-E60-Arandom-%0.25-P0.2-42",            # F1=0.6898
    "ro": "dumitrescustefan-C64-H3-E60-Aukn-%0.1-P0.2-42",    # F1=0.6758
}

# HuggingFace model IDs (for config.json)
MODEL_HF_IDS = {
    "FacebookAI-xlm-roberta-large": "FacebookAI/xlm-roberta-large",
    "lcampillos":        "lcampillos/roberta-base-biomedical-clinical-es",
    "PlanTL-GOB-ES":     "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
    "dccuchile":         "dccuchile/bert-base-spanish-wwm-cased",
    "microsoft":         "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "dmis-lab":          "dmis-lab/biobert-v1.1",
    "emilyalsentzer":    "emilyalsentzer/Bio_ClinicalBERT",
    "ufal":              "ufal/robeczech-base",
    "UWB-AIR":           "UWB-AIR/czert-B-base-cased",
    "IVN-RIN":           "IVN-RIN/Italian-BioNLP",
    "dbmdz":             "dbmdz/bert-base-italian-cased",
    "Musixmatch":        "Musixmatch/umberto-commoncrawl-cased-v1",
    "KB":                "KB/bert-base-swedish-cased",
    "vesteinn":          "vesteinn/ScandiBERT",
    "CLTL":              "CLTL/MedRoBERTa.nl",
    "pdelobelle":        "pdelobelle/robbert-v2",
    "GroNLP":            "GroNLP/bert-base-dutch-cased",
    "dumitrescustefan":  "dumitrescustefan/bert-base-romanian-cased-v1",
    "xlm-roberta-base":  "FacebookAI/xlm-roberta-base",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_run_name(name: str) -> dict:
    m = re.match(
        r"^(.+?)-C(\d+)-H(\d+)-E(\d+)-A(\w+)-%([\d.]+)-P([\d.]+)-(\d+)$", name
    )
    if not m:
        return {"raw": name}
    return {
        "model_prefix":    m.group(1),
        "context_size":    int(m.group(2)),
        "heads":           int(m.group(3)),
        "epochs":          int(m.group(4)),
        "augmentation":    m.group(5),
        "percentage_tags": float(m.group(6)),
        "aug_prob":        float(m.group(7)),
        "seed":            int(m.group(8)),
    }


def get_hf_id(run_name: str) -> str:
    prefix = parse_run_name(run_name).get("model_prefix", run_name)
    return MODEL_HF_IDS.get(prefix, prefix)


def load_tsv_fast(path: str, target_entity: str) -> dict:
    """Reads a TSV and returns a dict: { 'doc_id': [(start, end), (start, end)] } filtered by entity type."""
    doc_spans = defaultdict(list)
    
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        
        # Clean up column names in case they have trailing spaces
        reader.fieldnames = [c.strip() for c in reader.fieldnames]
        
        for row in reader:
            # ONLY grab rows that match the target entity
            label = row.get("label", "").strip()
            if label != target_entity.upper():
                continue
                
            doc_id = str(row["filename"])
            # Handle the 'strat_span' typo gracefully
            start = int(row.get("start_span", row.get("strat_span", 0)))
            end = int(row["end_span"])
            doc_spans[doc_id].append((start, end))
            
    return dict(doc_spans)


def get_txt_path(dataset_root: str, lang: str, entity: str, doc_id: str) -> str:
    return os.path.join(
        dataset_root,
        f"MultiClinNER-{lang}",
        f"MultiClinNER-{lang}-test",
        f"MultiClinNER-{lang}-test-{entity}",
        "txt",
        f"{doc_id}.txt",
    )


def _read_single_file(args):
    """Helper function for multithreaded file reading."""
    doc_id, filepath = args
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return doc_id, f.read()
    return doc_id, None


def load_txt_cache(dataset_root: str, lang: str, entity: str, doc_ids: list, shared_cache: dict):
    """Loads missing text files from disk into the shared dictionary using multithreading."""
    missing = []
    
    # Only hit the disk for IDs we haven't already loaded
    doc_ids_to_load = [d for d in doc_ids if d not in shared_cache]
    
    if doc_ids_to_load:
        print(f"    ⚡ Fetching {len(doc_ids_to_load)} raw .txt files using 32 threads...")
        
        # Prepare the arguments for the threads
        tasks = [(doc_id, get_txt_path(dataset_root, lang, entity, doc_id)) for doc_id in doc_ids_to_load]
        
        # Fire up a pool of threads. (32 is generally a sweet spot for network I/O)
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            # executor.map automatically handles the threading, and we wrap it in tqdm for the progress bar
            results = list(tqdm(
                executor.map(_read_single_file, tasks), 
                total=len(tasks), 
                desc="Reading TXTs", 
                leave=False, 
                unit="file", 
                mininterval=5.0
            ))
            
        # Process the threaded results
        for doc_id, content in results:
            if content is not None:
                shared_cache[doc_id] = content
            else:
                shared_cache[doc_id] = None
                missing.append(doc_id)
                
    if missing:
        print(f"    ⚠️  Missing source .txt for {len(missing)} docs: {missing[:5]}"
              + (" ..." if len(missing) > 5 else ""))

def build_config(run_name: str, lang: str, sub_num: int,
                 notes: str = "", is_ensemble: bool = False,
                 ensemble_runs: list = None) -> dict:
    p = parse_run_name(run_name)
    cfg = {
        "team":            "BIT.UA",
        "model":           get_hf_id(run_name),
        "architecture":    "Transformer + CRF (Multi-Head)",
        "language":        lang,
        "labels":          "multi-label",
        "scheme":          "BIO",
        "input_strategy":  "sentence_window",
        "window_size":     p.get("context_size", 64),
        "max_length":      256,
        "epochs":          p.get("epochs", "N/A"),
        "batch_size":      16,
        "lr":              "2e-5",
        "seed":            p.get("seed", "N/A"),
        "augmentation":    p.get("augmentation", "N/A"),
        "aug_prob":        p.get("aug_prob", "N/A"),
        "percentage_tags": p.get("percentage_tags", "N/A"),
        "ensemble":        is_ensemble,
        "submission":      sub_num,
        "notes":           notes,
    }
    if ensemble_runs:
        cfg["ensemble_runs"] = ensemble_runs
    return cfg


def build_readme(lang: str, entity: str, sub_num: int,
                 description: str, run_names: list, config: dict) -> str:
    lines = [
        f"# Submission {sub_num} — {lang.upper()} / {entity}",
        "",
        f"**Team:** BIT.UA  ",
        f"**Language:** {lang.upper()}  ",
        f"**Entity type:** {entity}  ",
        f"**Submission type:** {description}  ",
        "",
        "## Runs included",
        "",
    ]
    for r in run_names:
        lines.append(f"- `{r}`")
    lines += [
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, indent=2),
        "```",
        "",
        "## File format",
        "",
        "TSV columns: `filename`, `label`, `start_span`, `end_span`, `text`",
        "",
    ]
    return "\n".join(lines)


def write_submission(output_root: str, lang: str, entity: str,
                     sub_num: int, rows: list, config: dict, readme: str):
    folder = os.path.join(output_root, lang.upper(), entity, f"submission_{sub_num}")
    os.makedirs(folder, exist_ok=True)
    
    tsv_path = os.path.join(folder, f"{lang}_{entity}_submission{sub_num}.tsv")
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["filename", "label", "start_span", "end_span", "text"])
        writer.writerows(rows)
        
    with open(os.path.join(folder, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        
    with open(os.path.join(folder, "README.md"), "w") as f:
        f.write(readme)
        
    print(f"    ✅ submission_{sub_num}: {len(rows)} rows → {folder}")


# ---------------------------------------------------------------------------
# Prediction loading
# ---------------------------------------------------------------------------

def load_all_predictions(predictions_root: str, lang: str, entity: str) -> dict:
    folder = os.path.join(predictions_root, LANG_FOLDER[lang])
    
    # Try finding entity-specific files first, otherwise just load all TSVs in the dir
    files  = glob.glob(os.path.join(folder, f"*_{entity}.tsv"))
    if not files:
        files = glob.glob(os.path.join(folder, "*.tsv"))
        
    if not files:
        print(f"  ⚠️  No TSVs found in {folder}")
        return {}
        
    result = {}
    print(f"  📂 {lang}/{entity}: Loading {len(files)} TSV files...")
    
    for f in tqdm(files, desc="Loading TSVs", leave=False, unit="file", mininterval=5.0):
        # Clean up model name
        model_name = os.path.basename(f).replace(f"_{entity}.tsv", "").replace(".tsv", "")
        # Pass the target entity into the loader so it ignores the other 2 labels
        result[model_name] = load_tsv_fast(f, target_entity=entity)
        
    return result


# ---------------------------------------------------------------------------
# Single-model & ensemble builders
# ---------------------------------------------------------------------------

def make_single(model_dict: dict, entity: str, dataset_root: str, lang: str, shared_cache: dict) -> list:
    doc_ids = list(model_dict.keys())
    
    # Pre-load any text files we haven't seen yet into the shared cache
    load_txt_cache(dataset_root, lang, entity, doc_ids, shared_cache)
    
    rows = []
    for doc_id, spans in tqdm(model_dict.items(), desc="Formatting Single", leave=False, unit="doc", mininterval=5.0):
        raw_text = shared_cache.get(doc_id) or ""
        for start, end in spans:
            txt = raw_text[start:end] if raw_text else ""
            rows.append([doc_id, entity.upper(), start, end, txt])
            
    return rows


def make_ensemble(model_dicts: list, entity: str, dataset_root: str, lang: str, shared_cache: dict) -> list:
    all_doc_ids = set()
    for d in model_dicts:
        all_doc_ids.update(d.keys())

    # Pre-load any text files we haven't seen yet into the shared cache
    load_txt_cache(dataset_root, lang, entity, list(all_doc_ids), shared_cache)
    
    rows = []
    for doc_id in tqdm(sorted(all_doc_ids), desc="Ensembling Docs", leave=False, unit="doc", mininterval=5.0):
        raw_text = shared_cache.get(doc_id) or " " * 1_000_000

        # O(1) lookups for speed
        entities_input = [{"span": d.get(doc_id, [])} for d in model_dicts]

        result = ensemble_entity_level(entities_input, raw_text)

        for span, txt in zip(result["span"], result["text"]):
            rows.append([doc_id, entity.upper(), span[0], span[1], txt])

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@click.command()
@click.option("--predictions_root", required=True,
              help="Root folder with per-language prediction subfolders (ES/, EN/, …)")
@click.option("--dataset_root", required=True,
              help="Root of MultiClinNER dataset (contains MultiClinNER-{lang}/ subfolders)")
@click.option("--output_root", default="./submissions",
              help="Where to write submission folders  [default: ./submissions]")
@click.option("--lang",   default=None,
              help="Process only this language (e.g. es). Default: all.")
@click.option("--entity", default=None,
              help="Process only this entity type (disease/procedure/symptom). Default: all.")
def main(predictions_root, dataset_root, output_root, lang, entity):

    langs    = [lang]   if lang   else LANGUAGES
    entities = [entity] if entity else ENTITIES

    for lg in langs:
        best_run  = BEST_LANG_RUN.get(lg)
        
        # 🚀 Initialize persistent text cache for the ENTIRE language
        lang_txt_cache = {}

        for ent in entities:
            print(f"\n{'='*64}")
            print(f"  {lg.upper()} / {ent}")
            print(f"{'='*64}")

            all_preds = load_all_predictions(predictions_root, lg, ent)
            if not all_preds:
                print(f"  ⏭️  Skipping — no prediction files found on disk")
                continue

            # ── Sub 1: best multilingual ──────────────────────────────────
            print(f"\n  [1] Best multilingual")
            if BEST_MULTILINGUAL_RUN in all_preds:
                df1  = make_single(all_preds[BEST_MULTILINGUAL_RUN], ent, dataset_root, lg, lang_txt_cache)
                cfg1 = build_config(BEST_MULTILINGUAL_RUN, 'MIXED', 1,
                                    notes="Best multilingual (FacebookAI/xlm-roberta-large, fixed config)")
                write_submission(output_root, lg, ent, 1, df1, cfg1,
                                 build_readme('MIXED', ent, 1,
                                              "Best multilingual — FacebookAI/xlm-roberta-large",
                                              [BEST_MULTILINGUAL_RUN], cfg1))
            else:
                print(f"    ⚠️  {BEST_MULTILINGUAL_RUN} not found on disk")

            # ── Sub 2: best lang-specific ─────────────────────────────────
            print(f"\n  [2] Best lang-specific ({best_run})")
            if best_run and best_run in all_preds:
                df2  = make_single(all_preds[best_run], ent, dataset_root, lg, lang_txt_cache)
                cfg2 = build_config(best_run, lg, 2,
                                    notes="Best language-specific model by validation F1")
                write_submission(output_root, lg, ent, 2, df2, cfg2,
                                 build_readme(lg, ent, 2,
                                              "Best language-specific model (highest val F1)",
                                              [best_run], cfg2))
            else:
                print(f"    ⚠️  {best_run} not found on disk")

           # =====================================================================
            # DYNAMIC ENSEMBLES (Submissions 3, 4, 5)
            # =====================================================================
            
            # Split into categories based purely on "xlm-roberta"
            xlm_roberta_preds = {k: v for k, v in all_preds.items() if "xlm-roberta" in k.lower()}
            rest_preds        = {k: v for k, v in all_preds.items() if "xlm-roberta" not in k.lower()}
            
            # ── Sub 3: ensemble xlm-roberta runs ───────────────────────────────
            print(f"\n  [3] Ensemble {len(xlm_roberta_preds)} xlm-roberta runs")
            if xlm_roberta_preds:
                df3  = make_ensemble(list(xlm_roberta_preds.values()), ent, dataset_root, lg, lang_txt_cache)
                rep  = list(xlm_roberta_preds.keys())[0]
                cfg3 = build_config(rep, 'MIXED', 3,
                                    notes=f"Entity-level ensemble of {len(xlm_roberta_preds)} xlm-roberta runs",
                                    is_ensemble=True,
                                    ensemble_runs=list(xlm_roberta_preds.keys()))
                cfg3["model"] = "ensemble-xlm-roberta"
                write_submission(output_root, lg, ent, 3, df3, cfg3,
                                 build_readme('MIXED', ent, 3,
                                              f"Entity-level ensemble — xlm-roberta ({len(xlm_roberta_preds)} models)",
                                              list(xlm_roberta_preds.keys()), cfg3))
            else:
                print(f"    ⚠️  No xlm-roberta runs on disk")

            # ── Sub 4: ensemble the rest (non xlm-roberta) ─────────────────────
            print(f"\n  [4] Ensemble {len(rest_preds)} rest (non xlm-roberta) runs")
            if rest_preds:
                df4  = make_ensemble(list(rest_preds.values()), ent, dataset_root, lg, lang_txt_cache)
                rep  = list(rest_preds.keys())[0]
                cfg4 = build_config(rep, lg, 4,
                                    notes=f"Entity-level ensemble of {len(rest_preds)} non xlm-roberta runs",
                                    is_ensemble=True,
                                    ensemble_runs=list(rest_preds.keys()))
                cfg4["model"] = "ensemble-rest"
                write_submission(output_root, lg, ent, 4, df4, cfg4,
                                 build_readme(lg, ent, 4,
                                              f"Entity-level ensemble — rest ({len(rest_preds)} models)",
                                              list(rest_preds.keys()), cfg4))
            else:
                print(f"    ⚠️  No non xlm-roberta runs on disk")

            # ── Sub 5: ensemble EVERYTHING ─────────────────────────────────────
            print(f"\n  [5] Ensemble ALL {len(all_preds)} runs")
            if all_preds:
                df5  = make_ensemble(list(all_preds.values()), ent, dataset_root, lg, lang_txt_cache)
                rep = list(all_preds.keys())[0]
                cfg5 = build_config(rep, lg, 5,
                                    notes=f"Entity-level ensemble of ALL {len(all_preds)} runs",
                                    is_ensemble=True,
                                    ensemble_runs=list(all_preds.keys()))
                cfg5["model"] = "ensemble-all"
                write_submission(output_root, lg, ent, 5, df5, cfg5,
                                 build_readme(lg, ent, 5,
                                              f"Entity-level ensemble — ALL runs ({len(all_preds)} models)",
                                              list(all_preds.keys()), cfg5))
            else:
                print(f"    ⚠️  No runs available for ensemble")



    print(f"\n\n🎉 Done!  Output written to: {output_root}/")
    print("Structure: {LANG}/{entity}/submission_{{1-5}}/")
    print("  Each folder: {lang}_{entity}_submissionN.tsv  +  config.json  +  README.md")


if __name__ == "__main__":
    main()
