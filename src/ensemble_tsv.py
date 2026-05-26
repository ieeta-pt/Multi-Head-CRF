import os
import glob
import click
import pandas as pd
from collections import defaultdict

# Import YOUR functions
from ensemble import ensemble_span_level, ensemble_entity_level

@click.command()
@click.argument("predictions_dir")
@click.option("--docs_dir", default="../dataset/processed_multilingual/documents-all")
@click.option("--strategy", type=click.Choice(['span', 'entity']), default='span')
@click.option("--output_file", default="ensemble_output.tsv")
def main(predictions_dir, docs_dir, strategy, output_file):
    print(f"🚀 Loading TSVs from {predictions_dir}")
    tsv_files = glob.glob(os.path.join(predictions_dir, "*.tsv"))
    
    if not tsv_files:
        print("❌ No TSVs found in that directory!")
        return
        
    print(f"📦 Found {len(tsv_files)} models to ensemble.")

    # 1. Load all TSV predictions into memory
    # Structure: doc_label_spans[doc_id][label][model_idx] = [(start, end), ...]
    doc_label_spans = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for m_idx, file in enumerate(tsv_files):
        df = pd.read_csv(file, sep="\t")
        for _, row in df.iterrows():
            doc_id = str(row["filename"])
            label = row["label"]
            start = int(row["strat_span"])
            end = int(row["end_span"])
            doc_label_spans[doc_id][label][m_idx].append((start, end))

    # 2. Apply your Ensemble Logic
    print(f"🧠 Applying {strategy.upper()}-level ensembling...")
    ensemble_data = []
    
    for doc_id, labels_dict in doc_label_spans.items():
        # Load the raw text to reconstruct the string (needed for your ensemble functions)
        txt_path = os.path.join(docs_dir, f"{doc_id}.txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        else:
            raw_text = " " * 1000000 # Fallback dummy text if txt file isn't found
        
        for label, models_dict in labels_dict.items():
            # Prepare the `entities` list exactly as your ensemble functions expect it
            entities = []
            
            # We MUST loop through all models (even if empty) so your `majority_vote` math is correct!
            for m_idx in range(len(tsv_files)): 
                spans = models_dict.get(m_idx, [])
                entities.append({"span": spans})
            
            # Call your chosen ensemble function
            if strategy == "span":
                result = ensemble_span_level(entities, raw_text)
            else:
                result = ensemble_entity_level(entities, raw_text)
            
            # Append the surviving spans to our final output
            for i, span in enumerate(result["span"]):
                ensemble_data.append({
                    "filename": doc_id,
                    "strat_span": span[0],
                    "end_span": span[1],
                    "label": label,
                    "code": "-",
                    "text": result["text"][i]
                })
    
    # 3. Save to final TSV
    ensemble_df = pd.DataFrame(ensemble_data)
    ensemble_df.to_csv(output_file, sep="\t", index=False)
    print(f"✅ Saved {len(ensemble_df)} highly confident entities to {output_file}")

if __name__ == '__main__':
    main()