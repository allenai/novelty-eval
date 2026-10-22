#!/usr/bin/env python3
import argparse
import os
import yaml
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

def get_idea_title(metadata: Dict, key: str, is_pointwise: bool) -> Optional[str]:
    if is_pointwise:
        return metadata.get('title') if isinstance(metadata, dict) else None
    
    try:
        int_key = int(key)
        idea_meta = metadata.get(int_key)
    except ValueError:
        idea_meta = None
        
    if idea_meta is None:
        idea_meta = metadata.get(key)
        
    return idea_meta.get('title') if isinstance(idea_meta, dict) else None

def parse_markdown_tables(content: str) -> List[Dict]:
    """Extract candidate info from the markdown sections."""
    # Split by candidate headers. Use ^### to ensure we only match selected candidates, 
    # and not skipped ones which use ####.
    sections = re.split(r'(?m)^### \d+\. \[(.*?)\]\(.*?\)', content)
    candidate_names = sections[1::2]
    candidate_bodies = sections[2::2]
    
    candidates = []
    for name, body in zip(candidate_names, candidate_bodies):
        data = {'title': name}
        # Find the table after the name
        table_match = re.search(r'\| Field \| Value \|\n\|---\|---\|\n(.*?)(?=\n\n|\n#|$)', body, re.DOTALL)
        if table_match:
            table_rows = table_match.group(1).strip().split('\n')
            for row in table_rows:
                parts = [p.strip() for p in row.split('|') if p.strip()]
                if len(parts) >= 2:
                    data[parts[0]] = parts[1]
        candidates.append(data)
    return candidates

def verify_retrieval(test_inputs_path: str, debug_dir: str, cutoff_date: Optional[str], min_score: float):
    print(f"Loading test inputs from {test_inputs_path}...")
    with open(test_inputs_path, 'r') as f:
        inputs = yaml.safe_load(f)
    
    cutoff_dt = datetime.strptime(cutoff_date, '%Y-%m-%d') if cutoff_date else None
    
    duplicates = []
    correctly_skipped = []
    cutoff_violations = []
    score_violations = []
    
    for problem_id, data in inputs.items():
        is_pointwise = 'idea' in data and 'ideas' not in data
        ideas = {'idea': data['idea']} if is_pointwise else data.get('ideas', {})
        metadata = data.get('metadata', {})
        
        for idea_key in ideas:
            title = get_idea_title(metadata, str(idea_key), is_pointwise)
            debug_file = os.path.join(debug_dir, str(problem_id), f"{idea_key}_retrieval.md")
            if not os.path.exists(debug_file):
                continue
                
            with open(debug_file, 'r') as f:
                content = f.read()
            
            # Check for correctly skipped duplicates
            skipped_match = re.search(r'The following (\d+) candidate\(s\) were \*\*removed\*\* because their title exactly matches', content)
            if skipped_match:
                count = int(skipped_match.group(1))
                correctly_skipped.append((problem_id, idea_key, count))

            candidates = parse_markdown_tables(content)
            
            for c in candidates:
                c_title = c.get('title', '')
                
                # 1. Duplicate check
                if title and c_title.strip().lower() == title.strip().lower():
                    duplicates.append((problem_id, idea_key, c_title))
                
                # 2. Cutoff check
                pub_date_str = c.get('Publication Date')
                if pub_date_str and cutoff_dt:
                    try:
                        pub_dt = datetime.strptime(pub_date_str, '%Y-%m-%d')
                        if pub_dt >= cutoff_dt:
                            cutoff_violations.append((problem_id, idea_key, c_title, pub_date_str))
                    except ValueError:
                        pass
                elif cutoff_dt:
                    # Try year fallback
                    year_str = c.get('Year')
                    if year_str:
                        try:
                            if int(year_str) > cutoff_dt.year:
                                cutoff_violations.append((problem_id, idea_key, c_title, f"Year {year_str}"))
                        except ValueError:
                            pass

                # 3. Relevance score check
                score_str = c.get('Relevance Score', '0')
                try:
                    score = float(score_str)
                    if score < min_score:
                        score_violations.append((problem_id, idea_key, c_title, score))
                except ValueError:
                    pass

    print("\n" + "="*50)
    print("RETRIEVAL VERIFICATION REPORT")
    print("="*50)
    
    if correctly_skipped:
        total_skipped = sum(c[2] for c in correctly_skipped)
        print(f"✅ Correctly skipped {total_skipped} duplicates across {len(correctly_skipped)} ideas.")
    else:
        print("ℹ️ No 'Skipped' sections found in debug logs. (This is normal if no duplicates were found or if they were missed by the filter).")

    if not duplicates:
        print("✅ No title duplicates found in Selected Candidates.")
    else:
        print(f"❌ Found {len(duplicates)} title duplicates in Selected Candidates (Missed by filter):")
        for pid, ik, t in duplicates:
            print(f"  - Problem {pid}, Idea {ik}: \"{t}\"")
            
    print("\nNote: If 'Correctly skipped' is 0 but you expect duplicates, check if you are hitting the cache.")
    print("      The global skipped_duplicates_log.md is only populated during active retrieval.")
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Verify retrieval quality and constraints.")
    parser.add_argument('--test_inputs', type=str, required=True)
    parser.add_argument('--debug_dir', type=str, required=True)
    parser.add_argument('--cutoff_date', type=str, help="Cutoff date (YYYY-MM-DD). Candidates should be BEFORE this.")
    parser.add_argument('--min_score', type=float, default=0.0, help="Minimum relevance score.")
    args = parser.parse_args()
    
    verify_retrieval(args.test_inputs, args.debug_dir, args.cutoff_date, args.min_score)

if __name__ == "__main__":
    main()
