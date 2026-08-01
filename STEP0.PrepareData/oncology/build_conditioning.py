"""
Build multi-hot / categorical conditioning vectors for tumor MASK diffusion
(STEP2) from already-canonicalized RadGPT patient-note JSONs.

Conditioning covers exactly four things, per user spec:
    1. is_primary_site  -- one bit PER DIAGNOSIS ENTRY in the note: 1 if
                            that diagnosis's organ == its primary_site
                            (i.e. this is the primary tumor), 0 if organ
                            differs from primary_site (i.e. a metastatic
                            site). Derived directly from
                            cancer_diagnosis.organ vs.
                            cancer_diagnosis.primary_site already present
                            in the note -- no external organ list needed.
    2. radiation         -- multi-hot over radiation_therapy[] fields
    3. treatment         -- multi-hot over systemic_therapy[] fields
    4. tumor_markers     -- per-marker TREND classification (rising / falling
                             / stable / insufficient_data), NOT raw values,
                             since dates/individual values are unreliable.

Values in the patient note are assumed ALREADY CANONICALIZED. The
canonicalization folder is used ONLY to build the fixed vocabulary
(dimensionality) for each multi-hot field via each file's
"canonical_groups" keys -- it is not reapplied to the note at runtime.

Usage:
    python build_conditioning.py \
        --canon-dir /path/to/canonicalization_jsons \
        --note patient_note.json \
        --out conditioning.json
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict


# ----------------------------------------------------------------------
# Canonicalization-folder loading (vocabulary only)
# ----------------------------------------------------------------------

def load_canon_folder(canon_dir):
    """
    Load every *.json file in canon_dir. Each file is expected to look like
    the example provided:
        {
          "Feature Key": "diagnoses[].cancer_diagnosis.organ.value",
          "Unique Values": [...],
          "canonical_groups": {"Adrenal gland": [...], ...},
          "canonical_mapping": {...},
          ...
        }

    Returns dict: feature_key -> sorted list of canonical category names
                  (i.e. list(canonical_groups.keys())), plus the raw record.
    """
    canon_dir = Path(canon_dir)
    out = {}
    for fp in sorted(canon_dir.glob("*.json")):
        try:
            data = json.loads(fp.read_text())
        except json.JSONDecodeError:
            print(f"  [skip] {fp.name}: not valid JSON")
            continue

        feature_key = data.get("Feature Key")
        groups = data.get("canonical_groups")
        if not feature_key or not groups:
            print(f"  [skip] {fp.name}: missing 'Feature Key' or 'canonical_groups'")
            continue

        vocab = sorted(k for k in groups.keys() if k != "Not provided")
        out[feature_key] = {"vocab": vocab, "file": fp.name}

    return out


def find_feature_key(canon_index, *substrings):
    """
    Find feature keys in canon_index whose string contains ALL given
    substrings (case-insensitive). Returns list of matching keys.
    """
    matches = []
    for fk in canon_index:
        low = fk.lower()
        if all(s.lower() in low for s in substrings):
            matches.append(fk)
    return matches


# Exact filename -> role mapping, based on the known Vocab_Mask_UCSF_CanonicalizeV3
# folder layout (filenames mirror "Feature Key" with [] replaced by __).
# This is used in preference to fuzzy substring matching when the file exists.
KNOWN_FILES = {
    "organ": "diagnoses__.cancer_diagnosis.organ.value.json",
    "metastatic_site": "diagnoses__.cancer_stage.metastatic_sites__.site.json",
    "radiation_modality": "diagnoses__.radiation_therapy__.modality.value.json",
    "radiation_target_site": "diagnoses__.radiation_therapy__.target_site.value.json",
    "treatment_therapy_type": "diagnoses__.systemic_therapy__.therapy_type.value.json",
    "treatment_regimen": "diagnoses__.systemic_therapy__.regimen.value.json",
    "treatment_drug": "diagnoses__.systemic_therapy__.agents__.drug.json",
    "tumor_marker_name": "diagnoses__.tumor_markers__.marker.json",
}


def find_feature_key_by_filename(canon_index, filename):
    """Look up a feature key by matching the source filename recorded in
    canon_index[fk]['file']. Returns the feature key string or None."""
    for fk, info in canon_index.items():
        if info["file"] == filename:
            return fk
    return None



def describe_canon_index(canon_index):
    print(f"Loaded {len(canon_index)} canonicalization file(s):")
    for fk, info in canon_index.items():
        print(f"  - {fk}  (from {info['file']}, {len(info['vocab'])} categories)")


# ----------------------------------------------------------------------
# Helpers to safely pull values out of the (already canonical) note
# ----------------------------------------------------------------------

def _val(d, *path, default=None):
    """Walk a dict path, unwrapping {'value': ..., 'confidence': ...} leaves."""
    cur = d
    for key in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return default
    if isinstance(cur, dict) and "value" in cur:
        return cur.get("value", default)
    return cur if cur is not None else default


def get_diagnoses(note):
    """Note may be the raw dict as pasted (top-level 'diagnoses') -- handle
    both a bare dict and one nested under e.g. 'patient_context' siblings."""
    return note.get("diagnoses", []) or []


# ----------------------------------------------------------------------
# 1. is_primary_site -- single bit for THE organ this mask/lesion is in
# ----------------------------------------------------------------------

def extract_is_primary_site(note, target_organs, case_insensitive=True):
    """
    Filters diagnoses[] down to entries whose organ is in `target_organs`
    (your fixed list of 9), and returns a single is_primary_site bit for
    the matching diagnosis: 1 if that diagnosis's organ == primary_site
    (this IS the primary tumor), 0 if organ != primary_site (metastatic
    site). Comparisons are case-insensitive by default.

    Expected case: exactly one diagnosis entry matches target_organs.
    If zero or more than one match, this is logged (printed) and:
        - 0 matches  -> returns None
        - >1 matches -> logs all matches, returns the bit from the FIRST
                        match (so downstream code still gets a usable
                        single bit rather than crashing), but the
                        ambiguity is surfaced for manual review.
    """
    if case_insensitive:
        target_set = {t.strip().lower() for t in target_organs}
        organ_in_targets = lambda o: o is not None and o.strip().lower() in target_set
    else:
        target_set = set(target_organs)
        organ_in_targets = lambda o: o is not None and o in target_set

    matches = []
    for i, dx in enumerate(get_diagnoses(note)):
        organ = _val(dx, "cancer_diagnosis", "organ")
        if organ_in_targets(organ):
            matches.append((i, dx, organ))

    if len(matches) == 0:
        print("  [is_primary_site] WARNING: no diagnosis entry found with "
              f"organ in target list {sorted(target_organs)}")
        return None

    if len(matches) > 1:
        found_organs = [m[2] for m in matches]
        print(f"  [is_primary_site] WARNING: {len(matches)} diagnosis "
              f"entries matched target organs (expected exactly 1): "
              f"{found_organs}. Using the first match.")

    _, dx, organ = matches[0]
    primary_site = _val(dx, "cancer_diagnosis", "primary_site")
    if primary_site is None:
        print(f"  [is_primary_site] WARNING: matched organ '{organ}' but "
              f"primary_site is missing on that diagnosis entry")
        return None

    if case_insensitive:
        return int(organ.strip().lower() == primary_site.strip().lower())
    return int(organ == primary_site)


# ----------------------------------------------------------------------
# 2. Radiation (multi-hot)
# ----------------------------------------------------------------------

def _set_flag(vocab_dict, value):
    """Set vocab_dict[value] = 1 with a case-insensitive fallback match,
    so minor casing drift between canon files and notes doesn't silently
    zero out a field. Exact match is tried first."""
    if value is None:
        return
    if value in vocab_dict:
        vocab_dict[value] = 1
        return
    low = value.lower()
    for k in vocab_dict:
        if k.lower() == low:
            vocab_dict[k] = 1
            return
    # no match in vocab -- value not in the known category space, ignore


def extract_radiation_multihot(note, vocab_modality=None, vocab_site=None):
    """
    Multi-hot over radiation_therapy[] modality and/or target_site.
    vocab_modality / vocab_site: list of canonical categories (from the
    canonicalization folder) defining the fixed output dimensions. If a
    vocab list isn't supplied for a sub-field, that sub-field is skipped.
    """
    out = {}
    if vocab_modality is not None:
        out["modality"] = {v: 0 for v in vocab_modality}
    if vocab_site is not None:
        out["target_site"] = {v: 0 for v in vocab_site}

    for dx in get_diagnoses(note):
        for rt in dx.get("radiation_therapy", []) or []:
            modality = _val(rt, "modality")
            site = _val(rt, "target_site")
            if "modality" in out:
                _set_flag(out["modality"], modality)
            if "target_site" in out:
                _set_flag(out["target_site"], site)

    return out


# ----------------------------------------------------------------------
# 3. Treatment (multi-hot)
# ----------------------------------------------------------------------

def extract_treatment_multihot(note, vocab_type=None, vocab_regimen=None,
                                vocab_drug=None):
    """
    Multi-hot over systemic_therapy[] therapy_type / regimen / agents[].drug.
    Any vocab left as None is skipped.
    """
    out = {}
    if vocab_type is not None:
        out["therapy_type"] = {v: 0 for v in vocab_type}
    if vocab_regimen is not None:
        out["regimen"] = {v: 0 for v in vocab_regimen}
    if vocab_drug is not None:
        out["drug"] = {v: 0 for v in vocab_drug}

    for dx in get_diagnoses(note):
        for tx in dx.get("systemic_therapy", []) or []:
            ttype = _val(tx, "therapy_type")
            regimen = _val(tx, "regimen")
            if "therapy_type" in out:
                _set_flag(out["therapy_type"], ttype)
            if "regimen" in out:
                _set_flag(out["regimen"], regimen)

            for agent in tx.get("agents", []) or []:
                drug = agent.get("drug")
                if "drug" in out:
                    _set_flag(out["drug"], drug)

    return out


# ----------------------------------------------------------------------
# 4. Tumor markers -- TREND, not raw values
# ----------------------------------------------------------------------

def _parse_date(s):
    if not s:
        return None
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if not m:
        return None
    return tuple(int(x) for x in m.groups())  # (Y, M, D) tuple sorts fine


def classify_trend(values, min_points=3, flat_tol=0.10):
    """
    values: chronologically-ordered list of floats.
    Classifies overall trajectory using a robust slope-sign heuristic
    (first-half mean vs second-half mean) rather than raw endpoints,
    since individual timestamped values are noisy/unreliable.

    Returns one of: 'rising', 'falling', 'stable', 'insufficient_data'
    """
    values = [v for v in values if v is not None]
    if len(values) < min_points:
        return "insufficient_data"

    half = len(values) // 2
    first_half = values[:half] if half > 0 else values[:1]
    second_half = values[-half:] if half > 0 else values[-1:]

    m1 = sum(first_half) / len(first_half)
    m2 = sum(second_half) / len(second_half)

    if m1 == 0:
        return "insufficient_data"

    rel_change = (m2 - m1) / abs(m1)
    if rel_change > flat_tol:
        return "rising"
    elif rel_change < -flat_tol:
        return "falling"
    else:
        return "stable"


def extract_tumor_marker_trends(note, marker_vocab=None, flat_tol=0.10):
    """
    Groups tumor_markers[] by canonical marker name, sorts chronologically
    (dates used ONLY for ordering, never as a feature), and classifies each
    marker's trend. marker_vocab: fixed list of canonical marker names to
    report on (others ignored); if None, reports every marker found.
    """
    by_marker = defaultdict(list)
    for dx in get_diagnoses(note):
        for tm in dx.get("tumor_markers", []) or []:
            name = tm.get("marker")
            val = tm.get("value")
            date = _parse_date(tm.get("date"))
            if name is None or val is None:
                continue
            by_marker[name].append((date, val))

    trends = {}
    markers_to_report = marker_vocab if marker_vocab is not None else list(by_marker.keys())

    for marker in markers_to_report:
        entries = by_marker.get(marker, [])
        entries.sort(key=lambda x: (x[0] is None, x[0]))  # undated last
        ordered_values = [v for _, v in entries]
        trends[marker] = classify_trend(ordered_values, flat_tol=flat_tol)

    return trends


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def build_conditioning(note, canon_index,
                        radiation_modality_key=None,
                        radiation_site_key=None,
                        treatment_type_key=None,
                        treatment_regimen_key=None,
                        treatment_drug_key=None,
                        marker_key=None):

    def vocab_for(fk):
        return canon_index[fk]["vocab"] if fk and fk in canon_index else None

    result = {
        "is_primary_site": extract_is_primary_site(note,["Gallbladder", "Prostate", "Bladder", "Stomach", "Colon", "Duodenum", "Uterus", "Esophagus", "Spleen"]),
        "radiation": extract_radiation_multihot(
            note,
            vocab_modality=vocab_for(radiation_modality_key),
            vocab_site=vocab_for(radiation_site_key),
        ),
        "treatment": extract_treatment_multihot(
            note,
            vocab_type=vocab_for(treatment_type_key),
            vocab_regimen=vocab_for(treatment_regimen_key),
            vocab_drug=vocab_for(treatment_drug_key),
        ),
        "tumor_markers": extract_tumor_marker_trends(
            note,
            marker_vocab=vocab_for(marker_key),
        ),
    }
    return result


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--canon-dir", required=True,
                     help="Folder of canonicalization JSON files (vocab source only)")
    ap.add_argument("--note", required=True,
                     help="Path to a single (already-canonicalized) RadGPT patient note JSON")
    ap.add_argument("--out", default=None,
                     help="Output path for conditioning JSON (default: print to stdout)")
    ap.add_argument("--list-keys", action="store_true",
                     help="Just print discovered feature keys from canon-dir and exit")
    args = ap.parse_args()

    canon_index = load_canon_folder(args.canon_dir)
    describe_canon_index(canon_index)

    if args.list_keys:
        return

    # Auto-detect likely feature keys by substring match. Print what was
    # found/guessed so it can be corrected if wrong -- multiple hits or
    # zero hits both need a human decision.
    def pick(*substrings, label=""):
        matches = find_feature_key(canon_index, *substrings)
        if len(matches) == 1:
            print(f"  [auto] {label}: using '{matches[0]}'")
            return matches[0]
        elif len(matches) == 0:
            print(f"  [auto] {label}: no match found (skipping)")
            return None
        else:
            print(f"  [auto] {label}: AMBIGUOUS matches {matches} -- skipping, set manually")
            return None

    print("\nResolving feature keys by known filename (falls back to fuzzy match)...")

    def resolve(role, *fuzzy_substrings):
        filename = KNOWN_FILES.get(role)
        fk = find_feature_key_by_filename(canon_index, filename) if filename else None
        if fk:
            print(f"  [exact] {role}: \'{fk}\' (from {filename})")
            return fk
        return pick(*fuzzy_substrings, label=role)

    radiation_modality_key = resolve("radiation_modality", "radiation", "modality")
    radiation_site_key = resolve("radiation_target_site", "radiation", "target_site")
    treatment_type_key = resolve("treatment_therapy_type", "systemic_therapy", "therapy_type")
    treatment_regimen_key = resolve("treatment_regimen", "systemic_therapy", "regimen")
    treatment_drug_key = resolve("treatment_drug", "agents", "drug")
    marker_key = resolve("tumor_marker_name", "tumor_markers", "marker")

    note = json.loads(Path(args.note).read_text())

    try:
        conditioning = build_conditioning(
            note, canon_index,
            radiation_modality_key=radiation_modality_key,
            radiation_site_key=radiation_site_key,
            treatment_type_key=treatment_type_key,
            treatment_regimen_key=treatment_regimen_key,
            treatment_drug_key=treatment_drug_key,
            marker_key=marker_key,
        )
    except Exception:
        print("\n" + "=" * 70)
        print("ERROR while building conditioning vector. Dumping raw "
              "tumor_markers from this note for debugging:")
        print("=" * 70)
        for i, dx in enumerate(get_diagnoses(note)):
            tm = dx.get("tumor_markers")
            print(f"\n-- diagnoses[{i}].tumor_markers --")
            print(json.dumps(tm, indent=2, default=str))
        print("=" * 70 + "\n")
        raise

    out_str = json.dumps(conditioning, indent=2)
    if args.out:
        Path(args.out).write_text(out_str)
        print(f"\nWrote conditioning vector to {args.out}")
    else:
        print("\n" + out_str)


if __name__ == "__main__":
    main()