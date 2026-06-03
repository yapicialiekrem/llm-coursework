#!/usr/bin/env bash
# Download datasets and the MEB Tarih textbook PDFs (9, 10, 11, 12. sınıf).
# WMT16 + TurkishMMLU are pulled at runtime via HF `datasets`.
set -euo pipefail

mkdir -p data/raw/meb_tarih

# ---- MEB Tarih ders kitapları (Google Drive — derstarih.com mirror, 2024-2025 müfredat) ----
declare -A PDFS=(
  ["9_sinif_tarih.pdf"]="1kxl1xaEeZbJlURCB5l4bQ_BqUpD6PM0A"
  ["10_sinif_tarih.pdf"]="1olLJcT1cbBfgLHtKZ5kje9kHwppwSYKR"
  ["11_sinif_tarih.pdf"]="1USphVZfSW9mpHb4ULlFuvCUZlE4x1FTZ"
  ["12_sinif_inkilap.pdf"]="1Pdn0-q_nLtVnf-N9S2eeySByBHnBDO15"
)

for filename in "${!PDFS[@]}"; do
    target="data/raw/meb_tarih/${filename}"
    file_id="${PDFS[$filename]}"
    if [ -f "$target" ] && [ -s "$target" ]; then
        echo "[skip] $target already exists ($(du -h "$target" | cut -f1))"
    else
        echo "[get] $filename (drive id $file_id)"
        gdown --id "$file_id" -O "$target"
    fi
done

echo "---"
echo "[ok] PDFs in data/raw/meb_tarih/:"
ls -lh data/raw/meb_tarih/

# ---- Warm HF caches (optional) ----
python - <<'PY'
import os
print(f"HF_HOME = {os.environ.get('HF_HOME', '(unset)')}")
from datasets import load_dataset
print("Prefetching WMT16 tr-en (validation split only, to verify access)...")
load_dataset("wmt16", "tr-en", split="validation")
print("Prefetching TurkishMMLU (History subset)...")
load_dataset("AYueksel/TurkishMMLU", "History")
print("OK — caches ready.")
PY
