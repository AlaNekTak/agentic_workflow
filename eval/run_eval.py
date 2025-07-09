"""
run_eval.py  - quick evaluation harness

$ python eval/run_eval.py  --set bioasq  --n 100  --k 5
"""

from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

import rouge   # pip install rouge
from bert_score import score as bert_score  # pip install bert_score
from tqdm import tqdm

# --- import your agent -------------------------------------------------
from app import build_agent

# ---------- CLI & paths -----------------------------------------------
PARENT = Path(__file__).resolve().parent
DATA   = PARENT / "eval"              # ⇒ eval datasets here
OUT    = PARENT / "eval"
OUT.mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--set", choices=["bioasq", "sciqa", "pubmedqa"], default="bioasq")
parser.add_argument("--n",   type=int, default=10, help="#questions (0 = all)")
parser.add_argument("--k",   type=int, default=5,  help="Recall@k cutoff")
args   = parser.parse_args()

# ---------- load dataset ----------------------------------------------
def load_bioasq(n: int | None) -> List[Dict]:
    """Return list of {id, question, ideal_answer, pmids[]}"""
    with open(DATA / "bioasq_sample.json", encoding="utf8") as f:
        raw = json.load(f)
    items = []
    for entry in raw[: n or None]:
        items.append({
            "qid"   : entry["id"],
            "q"     : entry["question"],
            "gold"  : entry["ideal_answer"][0],
            "pmids" : entry["documents"]
        })
    return items

def load_sciqa(n: int | None) -> List[Dict]:
    with open(DATA / "sciqa_dev.json", encoding="utf8") as f:
        raw = json.load(f)
    return [{
        "qid": r["id"],
        "q"  : r["question"],
        "gold": r["answer"],
        "pmids": []          # no docs provided
    } for r in raw[: n or None]]

dataset = load_bioasq(args.n) if args.set=="bioasq" else load_sciqa(args.n)
print(f"Loaded {len(dataset)} questions from {args.set}")

# ---------- build agent once ------------------------------------------
agent = build_agent()

# ---------- eval loop --------------------------------------------------
rouge_l_scores = []
bert_f1_scores = []
recall_hits    = 0
tool_hits      = 0

rows = []
for item in tqdm(dataset, desc="Running"):
    q = item["q"]
    resp = agent.invoke({"messages":[{"role":"user","content": q}]})
    answer      : str           = resp["output"]["answer"]
    trace_buf  : list[str]      = resp["output"]["traces"]

    
    tool_used = "run_web_graph" if "run_web_graph" in "".join(trace_buf) \
                else "run_local_rag" if "run_local_rag" in "".join(trace_buf) \
                else "run_sql_graph" if "run_sql_graph" in "".join(trace_buf) else "none"
    
    # ---------- metrics -------------
    rl = rouge.Rouge(metrics=["rouge-l"])
    rouge_l = rl.get_scores(answer, item["gold"])[0]["rouge-l"]["f"] # type: ignore
    rouge_l_scores.append(rouge_l)

    P, R, F1 = bert_score([answer], [item["gold"]], lang="en", verbose=False)
    bert_f1_scores.append(F1[0].item())

    # crude PMID extraction
    retrieved_pmids = [pmid for pmid in item["pmids"] 
                       if any(pmid in t for t in trace_buf)]
    if retrieved_pmids:
        recall_hits += 1

    if args.set=="bioasq":
        correct_tool = "run_web_graph"  # BioASQ requires external search
        tool_hits += int(tool_used == correct_tool)
    
    rows.append({
        **item,
        "answer": answer,
        "rougeL": rouge_l,
        "bertF1": F1[0].item(),
        "tool"  : tool_used,
    })

# ---------- aggregate --------------------------------------------------
def mean(xs): return sum(xs)/len(xs) if xs else 0
report = {
    "N"          : len(dataset),
    "ROUGE-L"    : round(mean(rouge_l_scores), 4),
    "BERTScoreF1": round(mean(bert_f1_scores), 4),
    f"Recall@{args.k}" : round(recall_hits/len(dataset), 4),
    "Tool_Acc"   : round(tool_hits/len(dataset), 4) if args.set=="bioasq" else "n/a"
}
print("\n==  Aggregate results ==")
for k,v in report.items():
    print(f"{k:12}: {v}")

# ---------- save CSV ---------------------------------------------------
ts = time.strftime("%Y%m%d-%H%M%S")
out_file = OUT / f"{args.set}_{ts}.csv"
with open(out_file, "w", newline="", encoding="utf8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print(f"\nDetailed results → {out_file}")
