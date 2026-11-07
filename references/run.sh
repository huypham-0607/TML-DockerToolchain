#!/usr/bin/env bash
# Builds one hand-made reference image and runs its CPU smoke test.
#
#   references/run.sh ipmix            # or: ffb, ep_bnp, uap, all
#   KEEP=1 references/run.sh ffb       # keep the image afterwards (default: keep)
#   CLEAN=1 references/run.sh ffb      # remove the image after the smoke test
#
# ffb, ep_bnp and uap: the Dockerfile clones the repository at the pinned commit itself.
# ipmix: the repository is cloned into references/.repos/ipmix (git-ignored) first, and that
# clone is the build context (its Dockerfile uses COPY .).
# Repository URLs and pinned commits come from references/repos.txt.
# Extra `docker build` arguments can go in BUILD_ARGS, e.g.
#   BUILD_ARGS="--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118" references/run.sh ipmix
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

declare -A URL COMMIT
NAMES=()
while read -r name url commit; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  URL[$name]=$url COMMIT[$name]=$commit NAMES+=("$name")
done < "$HERE/repos.txt"

run_one() {
  local name=$1 image="tml-ref/$1:latest" context
  [[ -n "${URL[$name]:-}" ]] || { echo "unknown reference: $name (choose: ${!URL[*]} all)"; exit 2; }

  if grep -q '^COPY \. ' "$HERE/$name/Dockerfile"; then
    # this Dockerfile copies a local clone (build context = the clone)
    context="$HERE/.repos/$name"
    echo "=== $name: clone ${URL[$name]} @ ${COMMIT[$name]:0:7}"
    if [[ ! -d "$context/.git" ]]; then
      git clone --quiet "${URL[$name]}" "$context"
    fi
    git -C "$context" fetch --quiet origin "${COMMIT[$name]}" 2>/dev/null || true
    git -C "$context" checkout --quiet "${COMMIT[$name]}"
  else
    # this Dockerfile clones the repository itself: nothing to prepare
    context="$HERE/$name"
  fi

  echo "=== $name: build $image"
  local start=$SECONDS
  # shellcheck disable=SC2086
  docker build ${BUILD_ARGS:-} -f "$HERE/$name/Dockerfile" -t "$image" "$context"
  echo "=== $name: built in $((SECONDS - start))s, size $(docker image inspect -f '{{.Size}}' "$image" | awk '{printf "%.1f GB", $1/1e9}')"

  echo "=== $name: smoke test"
  docker run --rm -i --network none "$image" python - < "$HERE/$name/smoke.py"

  if [[ -n "${CLEAN:-}" ]]; then docker rmi -f "$image" >/dev/null; fi
}

if [[ "${1:-}" == "all" ]]; then
  for n in "${NAMES[@]}"; do run_one "$n"; done
else
  run_one "${1:?usage: references/run.sh <ipmix|ffb|ep_bnp|uap|all>}"
fi
