#!/usr/bin/env python3
import sys
import pandas as pd
import re

def main(in_path: str, out_path: str):
    # Read with header; auto-detect separator (tabs or spaces)
    df = pd.read_csv(in_path, sep=None, engine="python")

    # Expected columns (case-insensitive friendly mapping)
    colmap = {c.lower(): c for c in df.columns}
    req = {
        "gene": next(k for k in colmap if "gene" in k and "name" in k),
        "tss": next(k for k in colmap if "transcription" in k or "tss" in k),
        "strand": next(k for k in colmap if "strand" in k),
        "chrom": next(k for k in colmap if "chrom" in k),
    }

    g = df[colmap[req["gene"]]].astype(str).str.strip().str.upper()
    tss = pd.to_numeric(df[colmap[req["tss"]]], errors="coerce").astype("Int64")
    strand_raw = df[colmap[req["strand"]]].astype(str).str.strip()
    chrom_raw = df[colmap[req["chrom"]]].astype(str).str.strip()

    # Map strand {1, +1, '+'} → '+', {-1, -'} → '-'
    def map_strand(s: str) -> str | None:
        if s in {"1", "+1", "+", "1.0"}:
            return "+"
        if s in {"-1", "-1.0", "-", "−1"}:  # include Unicode minus just in case
            return "-"
        return None

    strand = strand_raw.map(map_strand)

    # Normalize chromosomes to mm-style prefix
    def norm_chrom(c: str) -> str | None:
        c = c.upper().replace("CHR", "")
        if c == "MT":
            return "chrM"
        # Keep primary chromosomes only
        if re.fullmatch(r"(?:[1-9]|1\d|2[0-2])", c):
            return f"chr{c}"
        if c in {"X", "Y"}:
            return f"chr{c}"
        return None   # drop non-canonical/alt scaffolds

    chrom = chrom_raw.map(norm_chrom)

    cleaned = pd.DataFrame({
        "chrom": chrom,
        "position": tss,
        "gene": g,
        "strand": strand,
    }).dropna()

    # Cast position back to plain int
    cleaned["position"] = cleaned["position"].astype(int)

    # Sort in natural chromosome order
    def chrom_key(c: str) -> tuple[int, int]:
        if c == "chrM": return (100, 0)
        if c == "chrX": return (101, 0)
        if c == "chrY": return (102, 0)
        m = re.fullmatch(r"chr(\d+)", c)
        return (int(m.group(1)), 0) if m else (999, 0)

    cleaned = cleaned.sort_values(
        by=["chrom", "position"],
        key=lambda s: s.map(chrom_key) if s.name == "chrom" else s
    )

    # Write as space- or tab-separated 4-column text
    cleaned[["chrom", "position", "gene", "strand"]].to_csv(
        out_path, sep="\t", header=False, index=False
    )

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write(
            "Usage: clean_hg38_tss.py TSS_hg38.txt cleaned_TSS_hg38_mmstyle.txt\n"
        )
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
