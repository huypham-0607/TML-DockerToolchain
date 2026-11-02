#!/usr/bin/env bash
# Builds one hand-made reference image and runs its CPU smoke test.
#
#   references/run.sh ipmix            # or: ffb, ep_bnp, uap, all
#   KEEP=1 references/run.sh ffb       # keep the image afterwards (default: keep)
#   CLEAN=1 references/run.sh ffb      # remove the image after the smoke test
#
# The repository is cloned at the pinned commit into references/.repos/<name> (git-ignored),
# then built with references/<name>/Dockerfile and that clone as the build context.
# Extra `docker build` arguments can go in BUILD_ARGS, e.g.
#   BUILD_ARGS="--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118" references/run.sh ipmix
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

declare -A URL=(
  [ipmix]=https://github.com/hzlsaber/IPMix
  [ffb]=https://github.com/ahxt/fair_fairness_benchmark
  [ep_bnp]=https://github.com/RJ-T/NIPS2022_EP_BNP
  [uap]=https://github.com/faiimea/UAP
)
declare -A COMMIT=(
  [ipmix]=050bcb6cb562e0f45b9caf64b17283ef58b21a41
  [ffb]=394f020e9d93cb195885ca5c67f411b175726e63
  [ep_bnp]=af5150da52fe55aa0e265352576c435e1d9a39ab
  [uap]=9f66dd0912b7c42ad3b74866036ff814a5609b0b
)

run_one() {
  local name=$1 repo="$HERE/.repos/$1" image="tml-ref/$1:latest"
  [[ -n "${URL[$name]:-}" ]] || { echo "unknown reference: $name (choose: ${!URL[*]} all)"; exit 2; }

  echo "=== $name: clone ${URL[$name]} @ ${COMMIT[$name]:0:7}"
  if [[ ! -d "$repo/.git" ]]; then
    git clone --quiet "${URL[$name]}" "$repo"
  fi
  git -C "$repo" fetch --quiet origin "${COMMIT[$name]}" 2>/dev/null || true
  git -C "$repo" checkout --quiet "${COMMIT[$name]}"

  echo "=== $name: build $image"
  local start=$SECONDS
  # shellcheck disable=SC2086
  docker build ${BUILD_ARGS:-} -f "$HERE/$name/Dockerfile" -t "$image" "$repo"
  echo "=== $name: built in $((SECONDS - start))s, size $(docker image inspect -f '{{.Size}}' "$image" | awk '{printf "%.1f GB", $1/1e9}')"

  echo "=== $name: smoke test"
  docker run --rm -i --network none "$image" python - < "$HERE/$name/smoke.py"

  if [[ -n "${CLEAN:-}" ]]; then docker rmi -f "$image" >/dev/null; fi
}

if [[ "${1:-}" == "all" ]]; then
  for n in ipmix ffb ep_bnp uap; do run_one "$n"; done
else
  run_one "${1:?usage: references/run.sh <ipmix|ffb|ep_bnp|uap|all>}"
fi
