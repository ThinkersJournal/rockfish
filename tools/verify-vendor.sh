#!/usr/bin/env bash
# Verify every vendored header against its upstream git blob.
#   tools/verify-vendor.sh            offline: hash our copy, compare with MANIFEST.tsv
#   tools/verify-vendor.sh --online   also ask GitHub for the blob sha at the pinned commit
# Exits non-zero on any mismatch or on a file that is on disk but not in the manifest.
set -euo pipefail
cd "$(dirname "$0")/.."
online=0; [ "${1:-}" = "--online" ] && online=1
fail=0

# verify_set <vendor dir> <header dir> <upstream repo> <pinned commit> <upstream licence path> <licence blob sha>
verify_set() {
  local dir=$1 headers=$2 upstream=$3 commit=$4 lic_path=$5 lic_sha=$6
  local checked=0 local_path upstream_path sha file got up rel f lic
  while IFS=$'\t' read -r local_path upstream_path sha; do
    case "$local_path" in '#'*|'') continue;; esac
    file="$headers/$local_path"
    got=$(git hash-object "$file")
    if [ "$got" != "$sha" ]; then echo "MISMATCH $file: ours $got, manifest $sha"; fail=1; fi
    if [ $online = 1 ]; then
      up=$(MSYS_NO_PATHCONV=1 gh api "repos/$upstream/contents/$upstream_path?ref=$commit" --jq .sha)
      if [ "$up" != "$sha" ]; then echo "UPSTREAM DIFFERS $upstream_path: forge $up, manifest $sha"; fail=1; fi
    fi
    checked=$((checked+1))
  done < "$dir/MANIFEST.tsv"
  lic=$(git hash-object "$dir/LICENSE.md")
  [ "$lic" = "$lic_sha" ] || { echo "MISMATCH $dir/LICENSE.md: $lic"; fail=1; }
  if [ $online = 1 ]; then
    up=$(MSYS_NO_PATHCONV=1 gh api "repos/$upstream/contents/$lic_path?ref=$commit" --jq .sha)
    [ "$up" = "$lic_sha" ] || { echo "UPSTREAM DIFFERS $lic_path: forge $up, expected $lic_sha"; fail=1; }
  fi
  # nothing on disk outside the manifest
  for f in $(git ls-files "$headers"); do
    rel=${f#"$headers"/}
    grep -q "^$rel	" "$dir/MANIFEST.tsv" || { echo "NOT IN MANIFEST $f"; fail=1; }
  done
  echo "$dir: checked $checked headers + licence (online=$online)"
}

# HIP: licence blob 4d43ac8... is projects/hip/LICENSE.md and projects/clr/LICENSE.md at the commit
verify_set vendor/hip vendor/hip/include ROCm/rocm-systems 97f5574fe2fdc7bef44fb01545347912ee9f1779 \
  projects/hip/LICENSE.md 4d43ac8700191c7a688e99ca2a25685963a61bb2
verify_set vendor/rocblas vendor/rocblas/include/rocblas ROCm/rocm-libraries 3a2a9bc877edcf5decd36eeaf897a8c9b6017030 \
  projects/rocblas/LICENSE.md e41274cd9434ff82a2e89735a3bad406b80b2f96
# hipBLASLt: includes hipblas-common/hipblas-common.h from projects/hipblas-common; its LICENSE.md is the same blob 4d43ac8...
verify_set vendor/hipblaslt vendor/hipblaslt/include ROCm/rocm-libraries 3a2a9bc877edcf5decd36eeaf897a8c9b6017030   projects/hipblaslt/LICENSE.md 4d43ac8700191c7a688e99ca2a25685963a61bb2
echo "fail=$fail"
exit $fail
