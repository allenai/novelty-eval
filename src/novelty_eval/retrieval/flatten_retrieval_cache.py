import yaml
import json
import os
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path to allow imports from novelty_eval
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from novelty_eval.retrieval.retrieval_reports import (
    generate_cache_status_report,
    format_retrieval_debug_info
)

def flatten_cache(pairwise_yaml_path, pairwise_cache_path, pointwise_yaml_path, output_cache_path):
    print(f"Loading pairwise YAML from {pairwise_yaml_path}...")
    with open(pairwise_yaml_path, 'r') as f:
        pw_yaml = yaml.safe_load(f)

    print(f"Loading pairwise cache from {pairwise_cache_path}...")
    with open(pairwise_cache_path, 'r') as f:
        pw_cache = json.load(f)

    print(f"Loading pointwise YAML from {pointwise_yaml_path}...")
    with open(pointwise_yaml_path, 'r') as f:
        ptw_yaml = yaml.safe_load(f)

    # 1. Build a mapping from idea_text to its cache entry in the nested pairwise cache
    idea_text_to_cache = {}
    for problem_id, p_data in pw_yaml.items():
        pid_str = str(problem_id)
        if pid_str not in pw_cache:
            continue
        
        ideas_dict = p_data.get('ideas', {})
        for idea_id, idea_text in ideas_dict.items():
            iid_str = str(idea_id)
            if iid_str in pw_cache[pid_str]:
                # Standardize text for mapping
                clean_text = idea_text.strip()
                idea_text_to_cache[clean_text] = pw_cache[pid_str][iid_str]

    print(f"Mapped {len(idea_text_to_cache)} ideas from pairwise cache.")

    # 2. Map pointwise instance IDs to the cache entries using idea_text
    ptw_cache = {}
    missing = 0
    for inst_id, inst_data in ptw_yaml.items():
        idea_text = inst_data.get('idea', '').strip()
        if idea_text in idea_text_to_cache:
            ptw_cache[str(inst_id)] = idea_text_to_cache[idea_text]
        else:
            missing += 1

    print(f"Created flattened cache with {len(ptw_cache)} entries. {missing} instances missing cache entries.")

    # 3. Save the new flat cache
    output_cache_path = Path(output_cache_path)
    output_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_cache_path, 'w') as f:
        json.dump(ptw_cache, f, indent=2)
    print(f"Flattened cache saved to: {output_cache_path}")

    # 4. Generate debug artifacts and status report
    output_path = output_cache_path.parent
    debug_dir = output_path / "retrieval_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating debug artifacts in {debug_dir}...")
    for inst_id, entry in ptw_cache.items():
        idea_text = entry.get("text", "")
        # Pointwise usually doesn't have 'context' in the cache entry but might in YAML
        topic = ptw_yaml.get(inst_id, {}).get("context", "N/A")
        
        # Construct debug_data from cache entry
        debug_data = {
            "contributions": entry.get("contributions", {}),
            "search_queries_dict": entry.get("search_queries_dict", {}),
            "candidates": entry.get("candidates", [])
        }
        
        debug_file = debug_dir / f"{inst_id}_retrieval.md"
        with open(debug_file, "w") as f:
            f.write(format_retrieval_debug_info(str(inst_id), idea_text, topic, debug_data))

    print(f"Generating cache status report...")
    # Mock args for generate_cache_status_report (it uses some fields)
    class MockArgs:
        test_inputs = str(pointwise_yaml_path)
        llm_engine = "cached"
        top_k_candidates = "N/A"
        cutoff_date = "N/A"
        max_contributions = "N/A"
        n_queries = "N/A"
        use_semantic_scholar = "N/A"
        max_search_workers = "N/A"
        nr_examples = "N/A"
    
    generate_cache_status_report(ptw_cache, ptw_yaml, str(output_cache_path), MockArgs())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flatten a nested pairwise retrieval cache into a flat pointwise cache.")
    parser.add_argument("--pw-yaml", type=str, required=True, help="Path to original pairwise iclr_test_instances.yaml")
    parser.add_argument("--pw-cache", type=str, required=True, help="Path to original pairwise retrieval_cache.json")
    parser.add_argument("--ptw-yaml", type=str, required=True, help="Path to target pointwise iclr_pointwise_instances.yaml")
    parser.add_argument("--out-cache", type=str, required=True, help="Path to save the new flattened retrieval_cache.json")
    
    args = parser.parse_args()
    flatten_cache(args.pw_yaml, args.pw_cache, args.ptw_yaml, args.out_cache)
