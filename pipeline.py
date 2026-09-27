# -*- coding: utf-8 -*-
"""
Business Entity Resolution Pipeline - Fast version
===================================================
Blocking  : inverted index on normalised name tokens (per country).
Matching  : rapidfuzz token_sort_ratio on name + address boost.
Outputs   : output/candidate_pairs.tsv, output/matching_results.tsv
"""

import os, re, sys, unicodedata, argparse, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from collections import defaultdict
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
TRAIN    = os.path.join(BASE, "dataset", "train")
TEST     = os.path.join(BASE, "dataset", "test")
OUT      = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

# ── hyper-params ───────────────────────────────────────────────────────────────
MAX_CANDS_PER_S1  = 25    # cap candidates per S1 entity (keeps candidate_pairs lean)
MIN_SHARED_TOKENS = 2     # must share ≥ this many significant name tokens to be a candidate
MATCH_THRESHOLD   = 0.52  # weighted score threshold → final match
MAX_POSTINGS      = 5000  # skip tokens whose posting list is longer (too common = no signal)

# ── normalisation ──────────────────────────────────────────────────────────────
_LEGAL = re.compile(
    r'\b(llc|llp|ltd|limited|pvt|private|inc|incorporated|corp|corporation|'
    r'co|company|plc|gmbh|sarl|sas|bv|nv|ag|sa|sl|srl|'
    r'solutions|services|enterprises|group|holdings|international|'
    r'associates|consulting|technologies|tech|systems|india|us|france)\b',
    re.IGNORECASE)
_PUNCT = re.compile(r'[^\w\s]')
_WS    = re.compile(r'\s+')
_STOP  = {'the', 'a', 'an', 'of', 'and', 'or', 'in', 'at', 'to', 'for', 'by', 'on'}

def nfc(t): return unicodedata.normalize('NFC', t)

def clean(text: str) -> str:
    if not isinstance(text, str): return ''
    t = nfc(text).lower()
    t = _PUNCT.sub(' ', t)
    return _WS.sub(' ', t).strip()

def core_name(text: str) -> str:
    """Name stripped of legal suffixes + stop words → used for blocking keys."""
    t = clean(text)
    t = _LEGAL.sub(' ', t)
    t = _WS.sub(' ', t).strip()
    return ' '.join(w for w in t.split() if w not in _STOP and len(w) > 1)

def sig_tokens(name: str) -> list:
    """Significant tokens (len ≥ 3, not stopword) from core name."""
    return [w for w in core_name(name).split() if len(w) >= 3]

# ── I/O helpers ────────────────────────────────────────────────────────────────

def load(path):
    return pd.read_csv(path, sep='\t', dtype=str).fillna('')

# ── blocking ───────────────────────────────────────────────────────────────────

def build_inverted_index(df: pd.DataFrame) -> dict:
    """token → list of entity_ids  (capped at MAX_POSTINGS to skip high-freq tokens)"""
    idx = defaultdict(list)
    for row in df.itertuples(index=False):
        for tok in sig_tokens(row.business_name):
            lst = idx[tok]
            if len(lst) < MAX_POSTINGS:
                lst.append(row.entity_id)
    # drop tokens that hit the cap (too common, no discriminative power)
    return {k: v for k, v in idx.items() if len(v) < MAX_POSTINGS}

def find_candidates(s1_name: str, inv_idx: dict, max_cands=MAX_CANDS_PER_S1) -> set:
    toks = sig_tokens(s1_name)
    if not toks:
        return set()
    counts = defaultdict(int)
    for t in set(toks):
        for eid in inv_idx.get(t, []):
            counts[eid] += 1
    min_shared = max(1, min(MIN_SHARED_TOKENS, len(toks)))
    qualified = {eid: c for eid, c in counts.items() if c >= min_shared}
    if len(qualified) > max_cands:
        top = sorted(qualified, key=lambda e: -qualified[e])[:max_cands]
        return set(top)
    return set(qualified)

def block(s1: pd.DataFrame, cands: pd.DataFrame) -> dict:
    """Returns dict: s1_entity_id → set of candidate entity_ids."""
    print(f"  Building inverted index over {len(cands):,} records...", flush=True)
    inv_idx = build_inverted_index(cands)

    result = {}
    for row in tqdm(s1.itertuples(index=False), total=len(s1), desc="  Blocking", miniters=50000):
        result[row.entity_id] = find_candidates(row.business_name, inv_idx)
    return result

SKIP_COUNTRIES = {'india'}   # skip for now, add later

def block_by_country(s1, s2, s3) -> dict:
    """Run blocking per country group to avoid cross-country noise."""
    cand_all = pd.concat([s2, s3], ignore_index=True)
    all_countries = s1['country'].str.lower().unique()
    out = {}
    for ctry in all_countries:
        s1_c   = s1[s1['country'].str.lower() == ctry]
        if ctry in SKIP_COUNTRIES:
            print(f"\n[{ctry}]  SKIPPED — {len(s1_c):,} S1 entities will have empty candidates", flush=True)
            for eid in s1_c['entity_id']: out[eid] = set()
            continue
        cand_c = cand_all[
            (cand_all['country'].str.lower() == ctry) |
            (cand_all['country'].str.strip() == '')
        ]
        print(f"\n[{ctry}]  S1={len(s1_c):,}  Cands={len(cand_c):,}", flush=True)
        if len(cand_c) == 0:
            for eid in s1_c['entity_id']: out[eid] = set()
            continue
        out.update(block(s1_c, cand_c))
    # fill any missed S1 entities
    for eid in s1['entity_id']:
        out.setdefault(eid, set())
    return out

# ── matching ───────────────────────────────────────────────────────────────────

def score(s1r, cr) -> float:
    n1 = clean(s1r.get('business_name',''))
    n2 = clean(cr.get('business_name',''))
    if not n1 or not n2: return 0.0

    c1 = core_name(s1r.get('business_name',''))
    c2 = core_name(cr.get('business_name',''))

    name_ts   = fuzz.token_sort_ratio(n1, n2) / 100
    name_core = fuzz.token_sort_ratio(c1, c2) / 100 if c1 and c2 else 0
    name_part = fuzz.partial_ratio(n1, n2) / 100

    a1 = clean(s1r.get('business_address',''))
    a2 = clean(cr.get('business_address',''))
    addr = fuzz.token_sort_ratio(a1, a2) / 100 if a1 and a2 else 0

    return 0.40*name_ts + 0.25*name_core + 0.15*name_part + 0.20*addr

def match(s1, s2, s3, candidates, threshold=MATCH_THRESHOLD) -> dict:
    cand_lkp = {r['entity_id']: r for r in pd.concat([s2,s3]).to_dict('records')}
    s1_lkp   = {r['entity_id']: r for r in s1.to_dict('records')}
    results  = {}
    for s1_id in tqdm(s1['entity_id'], desc="Matching", miniters=50000):
        s1r   = s1_lkp[s1_id]
        cids  = candidates.get(s1_id, set())
        results[s1_id] = [c for c in cids if score(s1r, cand_lkp[c]) >= threshold if c in cand_lkp]
    return results

# ── write outputs ──────────────────────────────────────────────────────────────

def write_tsv(data: dict, s1_ids, col, path):
    rows = [{'source1_entity_id': eid, col: ','.join(sorted(data.get(eid,set())))} for eid in s1_ids]
    pd.DataFrame(rows).to_csv(path, sep='\t', index=False)
    print(f"  Wrote {len(rows):,} rows → {path}", flush=True)

# ── F0.5 evaluation ────────────────────────────────────────────────────────────

def f05(pred, gold, ids):
    sc = []
    for sid in ids:
        p = set(pred.get(sid, []))
        g = set(gold.get(sid, []))
        if not p and not g: sc.append(1.0); continue
        if not p or not g:  sc.append(0.0); continue
        tp = len(p & g)
        pr = tp/len(p); re = tp/len(g)
        if pr+re == 0: sc.append(0.0)
        else: sc.append(1.25*pr*re/(0.25*pr+re))
    return float(np.mean(sc))

# ── validate on train sample ───────────────────────────────────────────────────

def validate(n=3000):
    print(f"\n=== Validation on {n} train samples ===")
    s1 = load(os.path.join(TRAIN,'train_source1.tsv'))
    s2 = load(os.path.join(TRAIN,'train_source2.tsv'))
    s3 = load(os.path.join(TRAIN,'train_source3.tsv'))
    gt = load(os.path.join(TRAIN,'train_ground_truth.tsv'))

    s1s = s1.sample(n=min(n, len(s1)), random_state=42).reset_index(drop=True)
    sample_ids = set(s1s['entity_id'])

    gold = {}
    for _, row in gt[gt['source1_entity_id'].isin(sample_ids)].iterrows():
        gold[row['source1_entity_id']] = [x.strip() for x in row['matched_entity_ids'].split(',') if x.strip()]

    cands = block_by_country(s1s, s2, s3)

    # Recall ceiling
    all_gold = [g for gs in gold.values() for g in gs]
    covered  = sum(1 for s,gs in gold.items() for g in gs if g in cands.get(s,set()))
    print(f"Blocking recall ceiling : {covered}/{len(all_gold)} = {covered/max(1,len(all_gold)):.3f}")
    print(f"Avg candidates per S1   : {np.mean([len(v) for v in cands.values()]):.1f}")

    preds = match(s1s, s2, s3, cands)
    print(f"F0.5 on sample          : {f05(preds, gold, list(s1s['entity_id'])):.4f}")

# ── main ───────────────────────────────────────────────────────────────────────

def run():
    print("=== Loading test data ===")
    s1 = load(os.path.join(TEST,'test_source1.tsv'))
    s2 = load(os.path.join(TEST,'test_source2.tsv'))
    s3 = load(os.path.join(TEST,'test_source3.tsv'))
    print(f"S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}")

    print("\n=== Blocking ===")
    cands = block_by_country(s1, s2, s3)
    write_tsv(cands, s1['entity_id'], 'candidate_entity_ids', os.path.join(OUT,'candidate_pairs.tsv'))

    print("\n=== Matching ===")
    preds = match(s1, s2, s3, cands)
    write_tsv(preds, s1['entity_id'], 'matched_entity_ids', os.path.join(OUT,'matching_results.tsv'))
    print("\n✓ Done")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--validate-size', type=int, default=3000)
    args = ap.parse_args()
    if args.validate:
        validate(args.validate_size)
    else:
        run()
