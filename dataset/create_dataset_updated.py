import os
import shutil
import glob
import pandas as pd
import re
from difflib import SequenceMatcher
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_ROOT = "/home/id.aau.dk/kr75cs/MultiClinNer/Multi-Head-CRF/dataset"
RAW_DATA_DIR = os.path.join(DATASET_ROOT, "MultiClinAI-training_data_v1.1-260225/MultiClinNER")

OUTPUT_DIR = os.path.join(DATASET_ROOT, "processed_multilingual")
DOCS_ALL_DIR = os.path.join(OUTPUT_DIR, "documents-all")

TRAIN_DOC_COUNT = 830
ENTITIES = ["symptom", "disease", "procedure"]

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def realign_span(old_text, new_text, old_start, old_end, entity_string):
    """Attempts to find the new span of an entity using context similarity."""
    try:
        matches = list(re.finditer(re.escape(str(entity_string)), new_text))
    except: return None
    
    if not matches: return None
    
    # 1. Direct Match Check (Best Case)
    for m in matches:
        if m.start() == old_start and m.end() == old_end:
            return (m.start(), m.end())

    # 2. Contextual Realignment (Similarity Check)
    context_size = 30
    old_context = old_text[max(0, old_start - context_size) : min(len(old_text), old_end + context_size)]
    
    best_match, best_score = None, -1
    for m in matches:
        new_start, new_end = m.span()
        new_context = new_text[max(0, new_start - context_size) : min(len(new_text), new_end + context_size)]
        score = SequenceMatcher(None, old_context, new_context).ratio()
        if score > best_score:
            best_score, best_match = score, (new_start, new_end)
            
    return best_match if best_score > 0.6 else None

def main():
    os.makedirs(DOCS_ALL_DIR, exist_ok=True)
    print(f"🚀 Starting Optimized Adaptive Consolidation.\nOutput: {OUTPUT_DIR}\n")

    lang_dirs = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "MultiClinNER-*")))
    if not lang_dirs:
        print("❌ No language directories found. Check your RAW_DATA_DIR path.")
        return

    all_train_dfs, all_val_dfs, all_full_dfs = [], [], []

    for lang_dir in lang_dirs:
        lang = os.path.basename(lang_dir).split('-')[-1]
        print(f"\n{'-'*60}\nProcessing Language: {lang.upper()}\n{'-'*60}")

        lang_tsv_dir = os.path.join(OUTPUT_DIR, lang)
        lang_docs_dir = os.path.join(OUTPUT_DIR, f"documents-{lang}")
        os.makedirs(lang_tsv_dir, exist_ok=True)
        os.makedirs(lang_docs_dir, exist_ok=True)

        # 1. Load all TSVs into Memory
        dfs = {}
        original_total = 0
        all_doc_ids = set()
        for ent in ENTITIES:
            path = os.path.join(lang_dir, f"MultiClinNER-{lang}-train", f"MultiClinNER-{lang}-train-{ent}", "tsv", f"MultiClinNER-{lang}-train-{ent}.tsv")
            if os.path.exists(path):
                df = pd.read_csv(path, sep="\t", names=['filename', 'label', 'start', 'end', 'text'], skiprows=1)
                df['doc_id'] = df['filename'].apply(lambda x: str(x).split('-')[-1])
                dfs[ent] = df
                original_total += len(df)
                all_doc_ids.update(df['doc_id'].unique())

        if len(dfs) < 3:
            print(f"⚠️ Skipping {lang} due to missing folder(s).")
            continue

        # 2. Optimized Document Loop with tqdm
        final_lang_rows = []
        anchor_stats = {e: 0 for e in ENTITIES}
        total_lost = 0
        sorted_ids = sorted(list(all_doc_ids))

        for doc_id in tqdm(sorted_ids, desc=f"   Aligning {lang}", unit="doc"):
            txts = {}
            valid_doc = True
            for ent in ENTITIES:
                p = os.path.join(lang_dir, f"MultiClinNER-{lang}-train", f"MultiClinNER-{lang}-train-{ent}", "txt", f"MultiClinNER-{lang}-train-{ent}-{doc_id}.txt")
                if os.path.exists(p):
                    with open(p, 'r', encoding='utf-8-sig', newline='') as f:
                        txts[ent] = f.read()
                else:
                    valid_doc = False
                    break
            
            if not valid_doc: continue

            # Speed Hack: Identity Shortcut (skip similarity check if texts match)
            if txts['symptom'] == txts['disease'] == txts['procedure']:
                anchor_stats['disease'] += 1 
                for ent in ENTITIES:
                    sub_df = dfs[ent][dfs[ent]['doc_id'] == doc_id].copy()
                    sub_df['start_span'], sub_df['end_span'] = sub_df['start'], sub_df['end']
                    sub_df['filename'] = f"{lang}-train-{doc_id}"
                    # Use .to_dict('records') to avoid mixing Series and Dicts
                    final_lang_rows.extend(sub_df.to_dict('records'))
                best_text = txts['disease']
            else:
                # Text Divergence Logic - Multi-Anchor Experiment
                doc_variants = []
                for pot_anchor in ENTITIES:
                    anchor_text = txts[pot_anchor]
                    rows, lost = [], 0
                    for source_ent in ENTITIES:
                        source_df = dfs[source_ent][dfs[source_ent]['doc_id'] == doc_id]
                        for _, row in source_df.iterrows():
                            res = realign_span(txts[source_ent], anchor_text, int(row['start']), int(row['end']), row['text'])
                            if res:
                                nr = row.copy()
                                nr['start_span'], nr['end_span'] = res
                                nr['filename'] = f"{lang}-train-{doc_id}"
                                # CRITICAL FIX: Convert Series to dict to avoid AttributeError
                                rows.append(nr.to_dict())
                            else: lost += 1
                    doc_variants.append((pot_anchor, rows, lost))

                # Select Anchor with highest retention for THIS document
                best_anchor, best_rows, best_lost = max(doc_variants, key=lambda x: len(x[1]))
                anchor_stats[best_anchor] += 1
                final_lang_rows.extend(best_rows)
                total_lost += best_lost
                best_text = txts[best_anchor]

            # Save cleaned text file (indices now match the chosen best_text)
            fname = f"{lang}-train-{doc_id}.txt"
            for d in [DOCS_ALL_DIR, lang_docs_dir]:
                with open(os.path.join(d, fname), 'w', encoding='utf-8') as f:
                    f.write(best_text)

        # 3. Post-Processing & Splitting
        if not final_lang_rows:
            continue

        lang_full_df = pd.DataFrame(final_lang_rows)
        lang_full_df = lang_full_df.drop_duplicates(subset=['filename', 'label', 'start_span', 'end_span']).sort_values(['filename', 'start_span'])
        lang_full_df['ann_id'] = lang_full_df.groupby('filename').cumcount()
        lang_full_df = lang_full_df[['filename', 'label', 'start_span', 'end_span', 'text', 'ann_id']]

        unique_docs = lang_full_df['filename'].unique()
        train_docs = unique_docs[:TRAIN_DOC_COUNT]
        val_docs = unique_docs[TRAIN_DOC_COUNT:]

        train_df = lang_full_df[lang_full_df['filename'].isin(train_docs)]
        val_df = lang_full_df[lang_full_df['filename'].isin(val_docs)]

        # Save language variants
        lang_full_df.to_csv(os.path.join(lang_tsv_dir, f"full_{lang}.tsv"), sep='\t', index=False)
        train_df.to_csv(os.path.join(lang_tsv_dir, f"train_{lang}.tsv"), sep='\t', index=False)
        val_df.to_csv(os.path.join(lang_tsv_dir, f"val_{lang}.tsv"), sep='\t', index=False)

        # Statistics
        retention = (len(lang_full_df) / original_total) * 100
        print(f"   📊 {lang.upper()} Stats: Retention: {retention:.2f}% | Lost: {total_lost}")
        print(f"   ⚓ Wins: S:{anchor_stats['symptom']} D:{anchor_stats['disease']} P:{anchor_stats['procedure']}")

        all_train_dfs.append(train_df); all_val_dfs.append(val_df); all_full_dfs.append(lang_full_df)

    # 4. Mixed Multilingual Datasets
    print(f"\n{'='*60}\n🌍 CREATING GLOBAL MIXED DATASETS\n{'='*60}")
    mixed_dir = os.path.join(OUTPUT_DIR, "mixed")
    os.makedirs(mixed_dir, exist_ok=True)

    if all_train_dfs:
        pd.concat(all_train_dfs, ignore_index=True).to_csv(os.path.join(mixed_dir, "train_mixed.tsv"), sep='\t', index=False)
        pd.concat(all_val_dfs, ignore_index=True).to_csv(os.path.join(mixed_dir, "val_mixed.tsv"), sep='\t', index=False)
        pd.concat(all_full_dfs, ignore_index=True).to_csv(os.path.join(mixed_dir, "full_mixed.tsv"), sep='\t', index=False)

        print(f"   ✅ Done! Mixed rows: Train({sum(len(d) for d in all_train_dfs)}), Val({sum(len(d) for d in all_val_dfs)}), Full({sum(len(d) for d in all_full_dfs)})")
    else:
        print("   ❌ No data processed successfully.")

if __name__ == "__main__":
    main()