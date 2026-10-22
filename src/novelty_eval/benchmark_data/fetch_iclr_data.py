from pathlib import Path

import numpy as np
import openreview
import json
import os
import sys
import openai
import anthropic
import argparse
import concurrent.futures
import time
from tqdm import tqdm
from datetime import datetime

import toml
import yaml
from jinja2 import Environment, FileSystemLoader

# Add the parent directory to sys.path to allow importing logging_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from logging_utils import setup_logger

# Initialize logger
LOGGER = setup_logger(output_dir='..', console_level='INFO')

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_BLOCKLIST_FILE = Path(__file__).resolve().parent / "paper_blocklist.yaml"
PROMPT_ENV = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
EXTRACT_NOVELTY_SIGNALS_PROMPT_TEMPLATE = PROMPT_ENV.get_template("extract_novelty_signals.jinja2")


def is_anthropic_model(model_name):
    return model_name.lower().startswith("claude")


def parse_json_from_text(text):
    if not text:
        return None

    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def request_novelty_json(client, extraction_model, prompt, provider, max_retries=8):
    for attempt in range(max_retries):
        try:
            if provider == "anthropic":
                response = client.messages.create(
                    model=extraction_model,
                    max_tokens=3000,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = "\n".join(
                    block.text for block in response.content if getattr(block, "type", None) == "text"
                )
                return parse_json_from_text(response_text)

            response = client.chat.completions.create(
                model=extraction_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            return parse_json_from_text(response.choices[0].message.content)

        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
            if is_rate_limit and attempt < max_retries - 1:
                wait = 2 ** attempt
                LOGGER.warning(f"Rate limit hit, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                raise


def load_secrets():
    project_root = Path(__file__).resolve().parents[3]
    secrets_file = project_root / "secrets.toml"
    if secrets_file.exists():
        secrets = toml.load(secrets_file)
        if "anthropic" in secrets:
            os.environ["ANTHROPIC_API_KEY"] = secrets["anthropic"]
        if "openai_key" in secrets:
            os.environ["OPENAI_API_KEY"] = secrets["openai_key"]
        if "openreview_username" in secrets:
            os.environ["OPENREVIEW_USERNAME"] = secrets["openreview_username"]
        if "openreview_password" in secrets:
            os.environ["OPENREVIEW_PASSWORD"] = secrets["openreview_password"]


def fetch_or_load_data(output_file, year, input_json_file=None):
    # If input JSON file is provided, load from it
    if input_json_file:
        if not os.path.exists(input_json_file):
            raise FileNotFoundError(f"Input JSON file not found: {input_json_file}")
        LOGGER.info(f"Loading data from provided input file: {input_json_file}")
        with open(input_json_file, 'r') as f:
            paper_data = json.load(f)
        LOGGER.info(f"Total papers loaded from input file: {len(paper_data)}")
        return paper_data

    # Initialize the client
    client = openreview.api.OpenReviewClient(
        baseurl='https://api2.openreview.net',
        username=os.getenv("OPENREVIEW_USERNAME"),
        password=os.getenv("OPENREVIEW_PASSWORD")
    )

    venue_ids = [
        f'ICLR.cc/{year}/Conference',
        f'ICLR.cc/{year}/Conference/Rejected_Submission',
        f'ICLR.cc/{year}/Conference/Withdrawn_Submission',
    ]

    submissions = []
    for venue_id in venue_ids:
        LOGGER.info(f"Fetching submissions for {venue_id}...")
        submissions.extend((client.get_all_notes(content={'venueid': venue_id}, details='replies')))

    paper_data = []

    for sub in submissions:
        # 1. Get Abstract and Primary Area (from the Submission note)
        content = sub.content
        abstract = content.get('abstract', {}).get('value', 'No abstract')
        # Note: field names like 'primary_area' can vary by conference (e.g., 'keywords')
        primary_area = content.get('primary_area', {}).get('value', 'N/A')
        title = content.get('title', {}).get('value', 'Untitled')

        # Get the decision from venue field or decision replies
        decision = None
        venue = content.get('venue', {}).get('value', '')
        if venue:
            if 'Accept' in venue or 'Poster' in venue or 'Spotlight' in venue or 'Oral' in venue:
                decision = 'Accepted'
            else:
                decision = 'Unaccepted'

        # 2. Get Average Rating and Decision (from the Replies)
        ratings = []
        reviews = []
        for reply in sub.details['replies']:
            # Filter for Official Reviews
            if 'Official_Review' in reply['invitations'][0]:
                reviews.append(reply['content'])
                ratings.append(reply['content'].get('rating', {}).get('value'))
            # Check for Decision in replies
            if 'Decision' in reply['invitations'][0] and decision is None:
                decision_content = reply['content'].get('decision', {}).get('value', '')
                if decision_content:
                    if 'Accept' in decision_content:
                        decision = 'Accept'
                    elif 'Reject' in decision_content:
                        decision = 'Reject'
                    else:
                        decision = decision_content

        avg_rating = np.mean(ratings) if ratings else 0

        paper_data.append({
            'title': title,
            'average_rating': avg_rating,
            'primary_area': primary_area,
            'abstract': abstract,
            'decision': decision,
            'reviews': reviews
        })

    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(paper_data, f, indent=4)

    LOGGER.info(f"Saved {len(paper_data)} papers to {output_file}")

    LOGGER.info(f"Total papers: {len(paper_data)}")
    return paper_data


def group_papers_by_area(paper_data):
    LOGGER.info("Grouping papers by primary area...")
    papers_by_area = {}
    for paper in paper_data:
        area = paper.get('primary_area', 'N/A')
        if area not in papers_by_area:
            papers_by_area[area] = []
        papers_by_area[area].append(paper)
    return papers_by_area


def filter_top_bottom_papers(papers_by_area, percentile):
    filtered_data = {}
    total_filtered_data = 0

    LOGGER.info("\nStats per area:")
    for area, papers in papers_by_area.items():
        if not papers:
            continue

        ratings = [p['average_rating'] for p in papers]
        if not ratings:
            continue

        # Calculate percentiles
        p_low = np.percentile(ratings, percentile)
        p_high = np.percentile(ratings, 100 - percentile)

        bottom_papers = [p for p in papers if p['average_rating'] <= p_low]
        top_papers = [p for p in papers if p['average_rating'] >= p_high]

        LOGGER.info(f"Area: {area}")
        LOGGER.info(f"  Top {percentile}% count: {len(top_papers)}")
        LOGGER.info(f"  Bottom {percentile}% count: {len(bottom_papers)}")
        LOGGER.info(f"  Edge cases count: {len(top_papers) + len(bottom_papers)}")

        filtered_data[area] = {
            'top_papers': top_papers,
            'bottom_papers': bottom_papers,
            'stats': {
                'count': len(top_papers) + len(bottom_papers),
                'p_low_threshold': p_low,
                'p_high_threshold': p_high,
            }
        }
        total_filtered_data += len(top_papers) + len(bottom_papers)

    LOGGER.info(f"\nFiltered data: {total_filtered_data}")
    return filtered_data


def process_single_review(review, extraction_model, client, provider):
    if 'positive_novelty_signals' in review and 'negative_novelty_signals' in review and 'similar_papers_mentioned' in review:
        return

    strengths = review.get('strengths', {}).get('value', '')
    weaknesses = review.get('weaknesses', {}).get('value', '')

    if not (strengths and weaknesses):
        review['positive_novelty_signals'] = []
        review['negative_novelty_signals'] = []
        review['similar_papers_mentioned'] = []
        return

    prompt = EXTRACT_NOVELTY_SIGNALS_PROMPT_TEMPLATE.render(strengths_review=strengths,
                                                            weaknesses_review=weaknesses)
    response_json = None
    try:
        response_json = request_novelty_json(client, extraction_model, prompt, provider)
    except Exception as e:
        LOGGER.error(f"Error extracting strengths novelty: {e}")

    review['positive_novelty_signals'] = response_json.get('positive_novelty_snippets', []) if response_json else []
    review['negative_novelty_signals'] = response_json.get('negative_novelty_snippets', []) if response_json else []
    review['similar_papers_mentioned'] = response_json.get('similar_papers_mentioned', []) if response_json else []


def is_area_processed(area_data):
    for group in ['top_papers', 'bottom_papers']:
        for paper in area_data[group]:
            for review in paper['reviews']:
                if 'positive_novelty_signals' not in review or 'similar_papers_mentioned' not in review:
                    return False
    return True


def extract_novelty_signals(filtered_data, max_areas_to_process, extraction_model, nr_workers, max_papers=None):
    provider = "anthropic" if is_anthropic_model(extraction_model) else "openai"
    if provider == "anthropic":
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for Claude models.")
        client = anthropic.Anthropic(api_key=anthropic_api_key)
    else:
        client = openai.OpenAI()

    LOGGER.info(f"Using {provider} provider with model: {extraction_model}")

    total_areas = len(filtered_data)
    process_all = max_areas_to_process is None or max_areas_to_process < 0
    areas_limit = total_areas if process_all else min(max_areas_to_process, total_areas)

    LOGGER.info(f"Processing {areas_limit} out of {total_areas} areas ({nr_workers} workers)...")
    if max_papers:
        LOGGER.info(f"Limiting to maximum {max_papers} papers total")

    reviews_to_process = []
    areas_processed = 0
    papers_processed = 0

    for area, data in filtered_data.items():
        if not process_all and areas_processed >= areas_limit:
            break

        if is_area_processed(data):
            LOGGER.info(f"Skipping area {area} (already processed)")
            continue

        LOGGER.info(f"Processing area: {area}")

        for group in ['top_papers', 'bottom_papers']:
            for paper in data[group]:
                # Check if we've reached the max papers limit
                if max_papers and papers_processed >= max_papers:
                    LOGGER.info(f"Reached max papers limit ({max_papers})")
                    break

                for review in paper['reviews']:
                    reviews_to_process.append(review)

                papers_processed += 1

            # Break out of group loop if limit reached
            if max_papers and papers_processed >= max_papers:
                break

        # Break out of area loop if limit reached
        if max_papers and papers_processed >= max_papers:
            break

        areas_processed += 1

    LOGGER.info(f"Total papers to process: {papers_processed}")
    LOGGER.info(f"Total reviews to process: {len(reviews_to_process)}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=nr_workers) as executor:
        futures = [executor.submit(process_single_review, review, extraction_model, client, provider) for review in
                   reviews_to_process]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing reviews"):
            pass


def _load_blocked_titles() -> set[str]:
    """Return lower-cased paper titles from paper_blocklist.yaml."""
    if not _BLOCKLIST_FILE.exists():
        return set()
    with open(_BLOCKLIST_FILE) as f:
        data = yaml.safe_load(f) or {}
    return {t.lower() for t in data.get("blocked_titles", [])}


def save_clean_novelty_dataset(filtered_data, output_filename):
    blocked = _load_blocked_titles()
    clean_data = {}

    for area, data in filtered_data.items():
        clean_top_papers = []
        clean_bottom_papers = []

        # Process Top Papers
        for paper in data['top_papers']:
            if blocked and paper.get('title', '').lower() in blocked:
                LOGGER.info(f"Skipping blocklisted paper: {paper.get('title')}")
                continue
            has_positive = False
            has_negative = False
            for review in paper['reviews']:
                if review.get('positive_novelty_signals'):
                    has_positive = True
                if review.get('negative_novelty_signals'):
                    has_negative = True

            if has_positive and not has_negative:
                clean_top_papers.append(paper)

        # Process Bottom Papers
        for paper in data['bottom_papers']:
            if blocked and paper.get('title', '').lower() in blocked:
                LOGGER.info(f"Skipping blocklisted paper: {paper.get('title')}")
                continue
            has_positive = False
            has_negative = False
            for review in paper['reviews']:
                if review.get('positive_novelty_signals'):
                    has_positive = True
                if review.get('negative_novelty_signals'):
                    has_negative = True

            if has_negative and not has_positive:
                clean_bottom_papers.append(paper)

        if clean_top_papers or clean_bottom_papers:
            clean_data[area] = {
                'top_papers': clean_top_papers,
                'bottom_papers': clean_bottom_papers
            }

    with open(output_filename, 'w') as f:
        json.dump(clean_data, f, indent=4)
    LOGGER.info(f"Saved clean novelty dataset to {output_filename}")


def main():
    load_secrets()

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_areas_to_process', type=int, default=-1,
                        help='Maximum number of areas to process. Use -1 to process all areas.')
    parser.add_argument('--max_papers', type=int, default=None,
                        help='Maximum total number of papers to process (combined top + bottom). Useful for quick prompt experimentation.')
    parser.add_argument('--model', type=str, default='claude-opus-4-6')
    parser.add_argument('--percentile', type=float, default=10.0, help='Percentile for filtering top/bottom papers')
    parser.add_argument('--nr_workers', type=int, default=60, help='Number of workers for parallel LLM extraction')
    parser.add_argument('--year', type=int, default=2026, help='ICLR year to process')
    parser.add_argument('--output_dir', type=str, default='iclr_data', help='Base output directory')
    parser.add_argument('--input_json', type=str,
                        default="iclr_data/iclr_2026_data.json",
                        help='Path to pre-retrieved ICLR data JSON file. If provided, data will be loaded from this file instead of fetching from OpenReview.')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    LOGGER.info(f"Output directory: {output_dir}")

    output_file = os.path.join(output_dir, f'iclr_{args.year}_data.json')
    filtered_output_file = os.path.join(output_dir, f'iclr_{args.year}_top_bottom_{int(args.percentile)}_percent.json')

    paper_data = fetch_or_load_data(output_file, args.year, args.input_json)
    papers_by_area = group_papers_by_area(paper_data)
    filtered_data = filter_top_bottom_papers(papers_by_area, args.percentile)

    extract_novelty_signals(filtered_data, args.max_areas_to_process, args.model, args.nr_workers, args.max_papers)

    with open(filtered_output_file, 'w') as f:
        json.dump(filtered_data, f, indent=4)

    LOGGER.info(f"\nSaved grouped and filtered data to {filtered_output_file}")

    clean_output_file = os.path.join(output_dir,
                                     f'iclr_{args.year}_clean_novelty_dataset_{int(args.percentile)}_percent.json')
    save_clean_novelty_dataset(filtered_data, clean_output_file)


if __name__ == "__main__":
    main()
