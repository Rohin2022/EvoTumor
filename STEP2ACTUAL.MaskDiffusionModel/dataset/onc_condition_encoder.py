"""
Encodes conditioning JSON (as produced by build_conditioning.py) into a
fixed-order, fixed-width torch tensor suitable for onc_cond in Unet3D.

Design: vocabularies (which keys exist, in what order) are fixed once from
a REFERENCE conditioning dict (e.g. any one sample, since all samples share
the same canon-derived vocab) -- NOT re-derived per sample. This keeps
per-__getitem__ calls cheap and guarantees every sample produces a tensor
of identical shape/order, which is required for collation.

Typical usage in a Dataset:

    class MyDataset(torch.utils.data.Dataset):
        def __init__(self, ..., reference_conditioning_json):
            self.onc_encoder = OncConditionEncoder(reference_conditioning_json)
            ...

        def __getitem__(self, idx):
            cond_json = json.load(open(self.conditioning_paths[idx]))
            onc_cond = self.onc_encoder.encode(cond_json)  # torch.FloatTensor
            return {..., "onc_cond": onc_cond}

Then in the training loop:
    data = next(self.dl)
    onc_cond = data["onc_cond"]   # (B, total_onc_cond), ready for Unet3D
"""

import torch


# Tumor marker trend has a fixed 4-way categorical space regardless of
# which markers exist. "unknown" and "insufficient_data" are treated as
# distinct labels since one usually means "test not ordered" and the other
# "test ordered but not enough points to trend" -- collapse them yourself
# below if you'd rather not distinguish.
TREND_CATEGORIES = ["rising", "falling", "stable", "insufficient_data"]


class OncConditionEncoder:
    """
    Builds a fixed vocabulary/ordering from a reference conditioning dict,
    then encodes any conditioning dict sharing that vocabulary into a single
    concatenated float tensor.

    Vector layout (in order):
        [is_primary_site]                                  -- 1 dim
        [radiation.modality one-hot* ]                      -- len(vocab)
        [treatment.therapy_type one-hot*]                   -- len(vocab)
        [treatment.regimen one-hot*]                        -- len(vocab)
        [treatment.drug one-hot*]                            -- len(vocab), if present
        [tumor_markers: one 4-way one-hot PER marker]        -- 4 * n_markers

        (* "one-hot*" = actually multi-hot: each category's 0/1 flag is
        taken directly from the input dict, so >1 category can be active
        at once, e.g. multiple regimens given historically.)

    self.cardinalities: OrderedDict of {block_name: width}, in emission
    order -- this is exactly what you pass as `oncology_conditioning` to
    Unet3D (sum of values must equal total_onc_cond).
    """

    def __init__(self, reference_conditioning):
        rc = reference_conditioning

        # --- is_primary_site: single scalar, unknown/None -> 0 ---
        self.has_is_primary_site = "is_primary_site" in rc

        # --- radiation blocks (dynamically discovered under "radiation") ---
        # e.g. {"modality": {...}, "target_site": {...}} -- whichever
        # sub-fields exist in the reference dict.
        self.radiation_fields = {}  # field_name -> sorted list of category keys
        for field_name, field_dict in (rc.get("radiation") or {}).items():
            self.radiation_fields[field_name] = sorted(field_dict.keys())

        # --- treatment blocks (dynamically discovered under "treatment") ---
        self.treatment_fields = {}
        for field_name, field_dict in (rc.get("treatment") or {}).items():
            self.treatment_fields[field_name] = sorted(field_dict.keys())

        # --- tumor markers: fixed marker name list, each gets a 4-way trend one-hot ---
        self.marker_names = sorted((rc.get("tumor_markers") or {}).keys())

        # --- build cardinalities in emission order ---
        self.cardinalities = {}
        if self.has_is_primary_site:
            self.cardinalities["is_primary_site"] = 1
        for field_name, cats in self.radiation_fields.items():
            self.cardinalities[f"radiation.{field_name}"] = len(cats)
        for field_name, cats in self.treatment_fields.items():
            self.cardinalities[f"treatment.{field_name}"] = len(cats)
        if self.marker_names:
            self.cardinalities["tumor_markers"] = len(self.marker_names) * len(TREND_CATEGORIES)

        self.total_dim = sum(self.cardinalities.values())

    # ------------------------------------------------------------------

    def encode(self, conditioning, dtype=torch.float32):
        """
        Encode one conditioning dict into a 1D torch tensor of length
        self.total_dim, following the fixed vocab/order established at
        __init__ time. Missing/unknown values default to 0 (all-false)
        rather than raising, so a sample with sparser data than the
        reference still produces a correctly-shaped vector.
        """
        chunks = []

        # is_primary_site: single element, None/unknown -> 0
        if self.has_is_primary_site:
            val = conditioning.get("is_primary_site")
            bit = 1.0 if val == 1 else 0.0  # None, 0, or anything else -> 0
            chunks.append(torch.tensor([bit], dtype=dtype))

        # radiation.<field>: multi-hot over fixed category list
        radiation = conditioning.get("radiation") or {}
        for field_name, cats in self.radiation_fields.items():
            field_dict = radiation.get(field_name) or {}
            vec = torch.tensor(
                [float(field_dict.get(cat, 0) or 0) for cat in cats],
                dtype=dtype,
            )
            chunks.append(vec)

        # treatment.<field>: multi-hot over fixed category list
        treatment = conditioning.get("treatment") or {}
        for field_name, cats in self.treatment_fields.items():
            field_dict = treatment.get(field_name) or {}
            vec = torch.tensor(
                [float(field_dict.get(cat, 0) or 0) for cat in cats],
                dtype=dtype,
            )
            chunks.append(vec)

        # tumor_markers: per marker, 4-way one-hot over TREND_CATEGORIES.
        # A marker missing from `conditioning` (not present at all) or with
        # a trend value outside TREND_CATEGORIES is encoded as all-zero
        # (equivalent to "insufficient_data" being unset) -- adjust here if
        # you'd rather force those into the "insufficient_data" slot instead.
        if self.marker_names:
            marker_data = conditioning.get("tumor_markers") or {}
            marker_vec = torch.zeros(
                len(self.marker_names) * len(TREND_CATEGORIES), dtype=dtype
            )
            for i, marker in enumerate(self.marker_names):
                trend = marker_data.get(marker)
                if trend in TREND_CATEGORIES:
                    j = TREND_CATEGORIES.index(trend)
                    marker_vec[i * len(TREND_CATEGORIES) + j] = 1.0
            chunks.append(marker_vec)

        return torch.cat(chunks, dim=0)

    # ------------------------------------------------------------------

    def describe(self):
        """Human-readable summary of the vector layout, for sanity checking
        against oncology_conditioning passed to Unet3D."""
        lines = [f"Total onc_cond dim: {self.total_dim}", "Blocks (in order):"]
        offset = 0
        for name, width in self.cardinalities.items():
            lines.append(f"  [{offset:4d}:{offset+width:4d}]  {name}  (width={width})")
            offset += width
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Standalone sanity check / example
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    example = json.loads(sys.stdin.read()) if not sys.stdin.isatty() else None
    if example is None:
        print("Provide a conditioning JSON on stdin to test encoding.")
        sys.exit(0)

    encoder = OncConditionEncoder(example)
    print(encoder.describe())
    print()
    vec = encoder.encode(example)
    print(f"Encoded shape: {tuple(vec.shape)}")
    print(f"Nonzero indices: {vec.nonzero().flatten().tolist()}")