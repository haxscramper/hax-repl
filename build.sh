#!/usr/bin/env bash
set -euo pipefail

tool_name="hax-repl"
dist_dir="dist"
req_file="${dist_dir}/requirements.txt"
out_pex="${dist_dir}/${tool_name}.pex"

mkdir -p "${dist_dir}"
rm -f "${req_file}" "${out_pex}"

# 1) Build your project wheel into dist/
uv build --wheel

# 2) Export locked runtime dependencies from uv.lock
uv export --frozen --no-dev --format requirements-txt --no-hashes --no-emit-project -o "${req_file}"

# 3) Build a single-file PEX executable
uv run pex \
  -r "${req_file}" \
  "${dist_dir}"/*.whl \
  -c "${tool_name}" \
  --venv \
  --scie eager \
  --python-shebang '/usr/bin/env python' \
  -o "${out_pex}"

chmod +x "${out_pex}"

echo "Built: ${out_pex}"
cp "${dist_dir}/${tool_name}" ~/software/hax-repl
