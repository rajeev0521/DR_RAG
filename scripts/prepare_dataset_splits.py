"""Prepares frozen, stratified benchmark dataset splits for HotpotQA and QASPER."""

from __future__ import annotations

import io
import json
import random
import tarfile
from pathlib import Path
from typing import Any, Dict, List
import requests
from datasets import load_dataset


def prepare_hotpotqa_splits(
    out_dir: Path,
    n_dev: int = 150,
    n_test: int = 150,
    seed: int = 42,
) -> None:
    """
    Samples stratified (by question type) dev and test splits from HotpotQA validation set.
    """
    print(f"Loading HotpotQA validation set for stratification (seed={seed})...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    # Group indices by query type ('bridge' vs 'comparison')
    by_type: Dict[str, List[int]] = {"bridge": [], "comparison": []}
    for idx, item in enumerate(ds):
        q_type = item.get("type", "bridge")
        if q_type in by_type:
            by_type[q_type].append(idx)
        else:
            by_type["bridge"].append(idx)

    rng = random.Random(seed)
    rng.shuffle(by_type["bridge"])
    rng.shuffle(by_type["comparison"])

    half_dev = n_dev // 2
    half_test = n_test // 2

    dev_indices = by_type["bridge"][:half_dev] + by_type["comparison"][:half_dev]
    test_indices = by_type["bridge"][half_dev : half_dev + half_test] + by_type["comparison"][half_dev : half_dev + half_test]

    rng.shuffle(dev_indices)
    rng.shuffle(test_indices)

    def _convert(indices: List[int]) -> List[Dict[str, Any]]:
        docs = []
        for i in indices:
            item = ds[i]
            doc_id = f"hotpot_{item['id']}"

            # Assemble structured text with markdown section headings
            text_parts = []
            context_titles = item["context"]["title"]
            context_sentences = item["context"]["sentences"]
            title_to_sents = {}

            for title, sents in zip(context_titles, context_sentences):
                title_to_sents[title] = sents
                sents_text = " ".join(sents)
                text_parts.append(f"# {title}\n\n{sents_text}")
            full_text = "\n\n".join(text_parts)

            # Map supporting facts to exact sentences
            gold_sentences = []
            supp_facts_tuples = []
            for t, s_idx in zip(item["supporting_facts"]["title"], item["supporting_facts"]["sent_id"]):
                supp_facts_tuples.append([t, int(s_idx)])
                if t in title_to_sents and s_idx < len(title_to_sents[t]):
                    gold_sentences.append(title_to_sents[t][s_idx].strip())

            qa_pair = {
                "query_id": f"q_{item['id']}",
                "question": item["question"],
                "answer": item["answer"],
                "query_type": item.get("type", "bridge"),
                "level": item.get("level", "medium"),
                "supporting_facts": supp_facts_tuples,
                "gold_sentences": gold_sentences,
            }

            docs.append({
                "doc_id": doc_id,
                "title": f"HotpotQA Context: {item['question']}",
                "text": full_text,
                "qa_pairs": [qa_pair],
            })
        return docs

    dev_docs = _convert(dev_indices)
    test_docs = _convert(test_indices)

    hotpot_dir = out_dir / "hotpotqa"
    hotpot_dir.mkdir(parents=True, exist_ok=True)

    with open(hotpot_dir / "dev.json", "w", encoding="utf-8") as f:
        json.dump(dev_docs, f, indent=2)
    with open(hotpot_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_docs, f, indent=2)

    # Also set as primary splits in data/splits/
    with open(out_dir / "dev.json", "w", encoding="utf-8") as f:
        json.dump(dev_docs, f, indent=2)
    with open(out_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_docs, f, indent=2)

    print(f"HotpotQA splits generated: {len(dev_docs)} dev, {len(test_docs)} test.")


def prepare_qasper_splits(
    out_dir: Path,
    n_papers_dev: int = 50,
    n_papers_test: int = 50,
    seed: int = 42,
) -> None:
    """
    Downloads official QASPER dev set and splits into dev and test scientific papers.
    """
    print("Downloading official QASPER dataset from AllenAI...")
    url = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    tar = tarfile.open(fileobj=io.BytesIO(resp.content))
    dev_file = tar.extractfile("qasper-dev-v0.3.json")
    raw_data: Dict[str, Any] = json.load(dev_file)

    paper_keys = list(raw_data.keys())
    rng = random.Random(seed)
    rng.shuffle(paper_keys)

    dev_keys = paper_keys[:n_papers_dev]
    test_keys = paper_keys[n_papers_dev : n_papers_dev + n_papers_test]

    def _convert_qasper(keys: List[str]) -> List[Dict[str, Any]]:
        docs = []
        for p_id in keys:
            paper = raw_data[p_id]
            title = paper.get("title", f"Scientific Paper {p_id}")
            abstract = paper.get("abstract", "")

            text_parts = [f"# {title}\n\n## Abstract\n\n{abstract}"]
            for sec in paper.get("full_text", []):
                sec_name = sec.get("section_name", "Section")
                paras = sec.get("paragraphs", [])
                sec_text = "\n\n".join(p.strip() for p in paras if p.strip())
                if sec_text:
                    text_parts.append(f"## {sec_name}\n\n{sec_text}")

            full_text = "\n\n".join(text_parts)

            qa_pairs = []
            for q_idx, qa in enumerate(paper.get("qas", [])):
                q_text = qa.get("question", "")
                answers = qa.get("answers", [])
                if not answers:
                    continue

                ans_data = answers[0].get("answer", {})
                if ans_data.get("unanswerable", False):
                    continue

                # Collect answer text
                if ans_data.get("extractive_spans"):
                    ans_text = ", ".join(ans_data["extractive_spans"])
                elif ans_data.get("free_form_answer"):
                    ans_text = ans_data["free_form_answer"]
                elif ans_data.get("yes_no") is not None:
                    ans_text = "Yes" if ans_data["yes_no"] else "No"
                else:
                    ans_text = ""

                if not ans_text:
                    continue

                # Evidence texts
                evidence_list = [e.strip() for e in ans_data.get("evidence", []) if e.strip()]
                if not evidence_list:
                    continue

                qa_pairs.append({
                    "query_id": f"q_qasper_{p_id}_{q_idx}",
                    "question": q_text,
                    "answer": ans_text,
                    "query_type": "scientific_qa",
                    "gold_sentences": evidence_list,
                    "supporting_facts": evidence_list,
                })

            if qa_pairs:
                docs.append({
                    "doc_id": f"qasper_{p_id}",
                    "title": title,
                    "text": full_text,
                    "qa_pairs": qa_pairs,
                })

        return docs

    dev_docs = _convert_qasper(dev_keys)
    test_docs = _convert_qasper(test_keys)

    qasper_dir = out_dir / "qasper"
    qasper_dir.mkdir(parents=True, exist_ok=True)

    with open(qasper_dir / "dev.json", "w", encoding="utf-8") as f:
        json.dump(dev_docs, f, indent=2)
    with open(qasper_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_docs, f, indent=2)

    total_dev_q = sum(len(d["qa_pairs"]) for d in dev_docs)
    total_test_q = sum(len(d["qa_pairs"]) for d in test_docs)
    print(f"QASPER splits generated: {len(dev_docs)} papers ({total_dev_q} QAs) dev, {len(test_docs)} papers ({total_test_q} QAs) test.")


def write_readme(out_dir: Path) -> None:
    readme_content = """# Benchmark Dataset Splits

This directory contains frozen, reproducible benchmark splits for evaluating STP-RAG and its ablation variants.

## 1. HotpotQA (Primary Multi-Hop Benchmark)
- **Source**: `hotpotqa/hotpot_qa` (validation distractor set).
- **Sub-directory**: `data/splits/hotpotqa/` (and root `dev.json`, `test.json`).
- **Stratification**: 50% "bridge" queries, 50% "comparison" queries.
- **Random Seed**: 42.
- **Split Sizes**:
  - `dev.json`: 150 documents, 150 QA pairs.
  - `test.json`: 150 documents, 150 QA pairs.
- **Document Structure**:
  - Each document comprises the 10 context Wikipedia paragraphs from a HotpotQA query formatted with Markdown section headers (`# <Article Title>`).
  - `gold_sentences` / `supporting_facts`: Exact sentences marked as supporting facts in the ground truth.

## 2. QASPER (Scientific Papers Benchmark)
- **Source**: AllenAI QASPER official dataset (`qasper-train-dev-v0.3.tgz`).
- **Sub-directory**: `data/splits/qasper/`.
- **Domain**: NLP and AI research papers from arXiv.
- **Random Seed**: 42.
- **Split Sizes**:
  - `qasper/dev.json`: 50 scientific papers.
  - `qasper/test.json`: 50 scientific papers.
- **Document Structure**:
  - Full papers formatted with hierarchical Markdown headers (`# Title`, `## Abstract`, `## Section`, `### Subsection`).
  - `gold_sentences`: Exact evidence paragraphs verified by NLP researchers.
"""
    with open(out_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("README.md written to data/splits/README.md.")


def main():
    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    prepare_hotpotqa_splits(splits_dir, n_dev=150, n_test=150, seed=42)
    prepare_qasper_splits(splits_dir, n_papers_dev=50, n_papers_test=50, seed=42)
    write_readme(splits_dir)


if __name__ == "__main__":
    main()
