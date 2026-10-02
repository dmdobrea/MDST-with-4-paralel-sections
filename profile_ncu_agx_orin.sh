#!/usr/bin/env bash
set -euo pipefail

EXE="${1:-./mdst_batched_agx_orin}"
TIMING_CSV="${2:-mdst_batch_timing.csv}"
NCU_CSV="${3:-mdst_ncu_metrics.csv}"
MERGED_CSV="${4:-mdst_agx_orin_results.csv}"

B_VALUES=(1 32 128 256 512 1024 2048 3072 4096 6144 8192 12288 16384)

OCC_METRIC="sm__warps_active.avg.pct_of_peak_sustained_active"
SM_ACTIVE_METRIC="sm__cycles_active.avg.pct_of_peak_sustained_elapsed"
SM_TPUT_METRIC="sm__throughput.avg.pct_of_peak_sustained_elapsed"
METRICS="${OCC_METRIC},${SM_ACTIVE_METRIC},${SM_TPUT_METRIC}"

if [[ ! -x "$EXE" ]]; then
  echo "ERROR: executable not found or not executable: $EXE" >&2
  exit 1
fi

NCU_BIN="$(command -v ncu || true)"
if [[ -z "$NCU_BIN" ]]; then
  echo "ERROR: ncu (Nsight Compute CLI) not found in PATH." >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required." >&2
  exit 1
fi

if [[ "${USE_SUDO:-0}" == "1" ]]; then
  NCU=(sudo "$NCU_BIN")
else
  NCU=("$NCU_BIN")
fi

mkdir -p ncu_raw

echo "Nsight Compute executable: $NCU_BIN"
echo "Nsight Compute version:"
"${NCU[@]}" --version 2>&1 | head -n 3 || true
echo

echo "[1/3] Running timing/scalability benchmark..."
"$EXE" --all --csv "$TIMING_CSV"

echo "[2/3] Profiling each B with Nsight Compute..."
echo "B,achieved_occupancy_pct,sm_active_pct,sm_throughput_pct" > "$NCU_CSV"

for B in "${B_VALUES[@]}"; do
  echo "  NCU: B=$B"
  RAW="ncu_raw/ncu_B${B}.csv"

  set +e
  # LC_ALL=C is requested for reproducible numeric formatting, but the parser
  # below also accepts decimal-comma output produced by some NCU/locale builds.
  LC_ALL=C "${NCU[@]}" \
    --csv \
    --page raw \
    --launch-count 1 \
    --metrics "$METRICS" \
    "$EXE" --b "$B" --profile-only \
    > "$RAW" 2>&1
  STATUS=$?
  set -e

  if [[ $STATUS -ne 0 ]]; then
    echo "ERROR: ncu failed for B=$B. See $RAW" >&2
    if grep -qiE "permission|ERR_NVGPUCTRPERM|profiling.*restricted" "$RAW"; then
      echo "Hint: retry with:" >&2
      echo "  USE_SUDO=1 ./profile_ncu_agx_orin.sh $EXE" >&2
    fi
    exit $STATUS
  fi

  # NCU --page raw emits a WIDE CSV table:
  #   header row: metric names are columns
  #   units row
  #   data row
  # Profiler status lines may precede the CSV. Parse the table by matching
  # exact column names rather than grepping individual rows.
  python3 - "$RAW" "$B" "$OCC_METRIC" "$SM_ACTIVE_METRIC" "$SM_TPUT_METRIC" >> "$NCU_CSV" <<'PY'
import csv
import io
import sys

path, B, *metrics = sys.argv[1:]

with open(path, 'r', errors='replace') as f:
    csv_lines = [ln for ln in f if ln.lstrip().startswith('"')]

if not csv_lines:
    sys.stderr.write(f"ERROR: No CSV table found in {path} for B={B}\n")
    sys.exit(3)

rows = list(csv.reader(io.StringIO(''.join(csv_lines))))

header_idx = None
for i, row in enumerate(rows):
    if all(m in row for m in metrics):
        header_idx = i
        break

if header_idx is None:
    sys.stderr.write(f"ERROR: Required NCU metric columns not found for B={B}: {metrics}\n")
    sys.stderr.write(f"Inspect raw output: {path}\n")
    sys.exit(3)

header = rows[header_idx]
col = {m: header.index(m) for m in metrics}

# Skip the units row (its first field is empty) and select the first actual
# kernel record. In NCU raw CSV the first field is normally numeric ID=0.
data = None
for row in rows[header_idx + 1:]:
    if len(row) < len(header):
        row = row + [''] * (len(header) - len(row))
    first = row[0].strip() if row else ''
    if first and first.lstrip('+-').isdigit():
        data = row
        break

if data is None:
    sys.stderr.write(f"ERROR: NCU data row not found for B={B}\n")
    sys.stderr.write(f"Inspect raw output: {path}\n")
    sys.exit(3)

def parse_number(s: str) -> float:
    s = s.strip().strip('"').replace('%', '').replace('\u00a0', '')
    # Target metrics are percentages, so no thousands separator is expected.
    # Accept both decimal comma (e.g. 5,22) and decimal point (5.22).
    if ',' in s and '.' not in s:
        s = s.replace(',', '.')
    elif ',' in s and '.' in s:
        # Conservative handling for locale-formatted values. If comma occurs
        # after the last dot, interpret comma as decimal separator and dots as
        # grouping separators; otherwise treat commas as grouping separators.
        if s.rfind(',') > s.rfind('.'):
            s = s.replace('.', '').replace(',', '.')
        else:
            s = s.replace(',', '')
    return float(s)

vals = {}
for m in metrics:
    try:
        vals[m] = parse_number(data[col[m]])
    except Exception as e:
        sys.stderr.write(
            f"ERROR: Could not parse {m!r} value {data[col[m]]!r} "
            f"for B={B}: {e}\n"
        )
        sys.exit(3)

print(B + ',' + ','.join(f'{vals[m]:.6f}' for m in metrics))
PY

done

echo "[3/3] Merging timing and NCU metrics..."
python3 - "$TIMING_CSV" "$NCU_CSV" "$MERGED_CSV" <<'PY'
import csv
import sys

timing_path, ncu_path, out_path = sys.argv[1:]

with open(timing_path, newline='') as f:
    timing = list(csv.DictReader(f))
with open(ncu_path, newline='') as f:
    ncu = {row['B']: row for row in csv.DictReader(f)}

if not timing:
    raise RuntimeError('Timing CSV is empty')

extra = ['achieved_occupancy_pct', 'sm_active_pct', 'sm_throughput_pct']
fields = list(timing[0].keys()) + extra

with open(out_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for row in timing:
        nr = ncu.get(row['B'])
        if nr is None:
            raise RuntimeError(f"Missing NCU row for B={row['B']}")
        out = dict(row)
        for k in extra:
            out[k] = nr[k]
        w.writerow(out)
PY

echo
echo "Done."
echo "  Timing results : $TIMING_CSV"
echo "  NCU metrics    : $NCU_CSV"
echo "  Merged table   : $MERGED_CSV"
echo "  Raw NCU output : ncu_raw/ncu_B*.csv"
