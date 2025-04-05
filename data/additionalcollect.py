from functools import partial
import os
import datasets
from tqdm import tqdm

# Directory structure
DATA_ROOT = "/root/nanogpt/datasets"
DATASET_DIR = os.path.join(DATA_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")

# Create directories if they don't exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_curated_thoughts():
    # Load all three configurations
    configs = [
        'OpenR1-Math-220k-default',
        'OpenThoughts-114k-math-default',
        'OpenThoughts-114k-metadata'
    ]
    
    datasets_dict = {}
    for config in configs:
        ds_dict = datasets.load_dataset(
            "bethgelab/CuratedThoughts", 
            config,
            trust_remote_code=True, 
            cache_dir=DATASET_DIR
        )
        # Get the first split (usually 'train' or 'default')
        ds = list(ds_dict.values())[0]
        
        ds = ds.map(lambda x: {
            "problem": x.pop("problem", ""),
            "solution": x.pop("solution", ""),
            "generations": x.pop("generations", ""),
            "answer": x.pop("answer", ""),
            "source": f"bethgelab/CuratedThoughts/{config}",
            "metadata": str(x)
        })
        datasets_dict[config] = ds
    
    return datasets_dict

def load_wildchat():
    ds_dict = datasets.load_dataset(
        "bigstupidhats/wildchat_conversations",
        trust_remote_code=True,
        cache_dir=DATASET_DIR
    )
    # Get the first split
    ds = list(ds_dict.values())[0]
    
    # Split into English and Russian based on language detection
    def is_english(text):
        # Simple heuristic: if more than 50% of characters are ASCII, consider it English
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        return ascii_chars / len(text) > 0.5 if text else True

    def process_conversation(example):
        # Get all available fields with empty string defaults
        conversation = str(example.get('conversation', ''))
        instruction = str(example.get('instruction', ''))
        output = str(example.get('output', ''))
        
        # Determine language
        text_to_check = ' '.join([conversation, instruction, output])
        language = "english" if is_english(text_to_check) else "russian"
        
        return {
            "conversation": conversation,
            "instruction": instruction,
            "output": output,
            "language": language,
            "source": "bigstupidhats/wildchat_conversations",
            "metadata": str(example)
        }
    
    ds = ds.map(process_conversation)
    
    # Split into two datasets based on language
    ds_en = ds.filter(lambda x: x['language'] == 'english')
    ds_ru = ds.filter(lambda x: x['language'] == 'russian')
    
    return {"english": ds_en, "russian": ds_ru}

if __name__ == "__main__":
    # Dictionary mapping dataset names to their load functions
    datasets_to_load = {
        "CuratedThoughts": load_curated_thoughts,
        "WildChat": load_wildchat,
    }
    
    # Load and save each dataset
    for name, load_fn in tqdm(datasets_to_load.items(), desc="Processing datasets"):
        print(f"\nProcessing {name}...")
        
        try:
            ds = load_fn()
            
            if name == "CuratedThoughts":
                for config, split_ds in ds.items():
                    output_path = os.path.join(PROCESSED_DIR, f"{name}_{config}.parquet")
                    columns_to_keep = ['problem', 'solution', 'generations', 'answer', 'source', 'metadata']
                    split_ds = split_ds.select_columns(columns_to_keep)
                    split_ds.to_parquet(output_path)
                    print(f"Saved {name} {config} split to {output_path}")
            
            elif name == "WildChat":
                for lang, split_ds in ds.items():
                    output_path = os.path.join(PROCESSED_DIR, f"{name}_{lang}.parquet")
                    columns_to_keep = ['conversation', 'instruction', 'output', 'language', 'source', 'metadata']
                    split_ds = split_ds.select_columns(columns_to_keep)
                    split_ds.to_parquet(output_path)
                    print(f"Saved {name} {lang} split to {output_path}")
                
        except Exception as e:
            print(f"Error processing {name}: {str(e)}")
            continue

    print("\nAll datasets processed and saved!")
