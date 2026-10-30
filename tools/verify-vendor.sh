#!/usr/bin/env bash
# Verify every vendored HIP header against its upstream git blob.
#   tools/verify-vendor.sh            offline: hash our copy, compare with MANIFEST.tsv
#   tools/verify-vendor.sh --online   also ask GitHub for the blob sha at the pinned commit
# Exits non-zero on any mismatch or on a file that is on disk but not in the manifest.
set -euo pipefail
cd "$(dirname "$0")/.."
UPSTREAM=ROCm/rocm-systems
COMMIT=97f5574fe2fdc7bef44fb01545347912ee9f1779
online=0; [ "${1:-}" = "--online" ] && online=1
fail=0; checked=0
while IFS=$'\t' read -r local upstream sha; do
  case "$local" in '#'*|'') continue;; esac
  file="vendor/hip/include/$local"
  got=$(git hash-object "$file")
  if [ "$got" != "$sha" ]; then echo "MISMATCH $file: ours $got, manifest $sha"; fail=1; fi
  if [ $online = 1 ]; then
    up=$(MSYS_NO_PATHCONV=1 gh api "repos/$UPSTREAM/contents/$upstream?ref=$COMMIT" --jq .sha)
    if [ "$up" != "$sha" ]; then echo "UPSTREAM DIFFERS $upstream: forge $up, manifest $sha"; fail=1; fi
  fi
  checked=$((checked+1))
done < vendor/hip/MANIFEST.tsv
# licence text: blob 4d43ac8... is projects/hip/LICENSE.md and projects/clr/LICENSE.md at the commit
lic=$(git hash-object vendor/hip/LICENSE.md)
[ "$lic" = 4d43ac8700191c7a688e99ca2a25685963a61bb2 ] || { echo "MISMATCH vendor/hip/LICENSE.md: $lic"; fail=1; }
# nothing on disk outside the manifest
for f in $(git ls-files vendor/hip/include); do
  rel=${f#vendor/hip/include/}
  grep -q "^$rel	" vendor/hip/MANIFEST.tsv || { echo "NOT IN MANIFEST $f"; fail=1; }
done
echo "checked $checked headers + licence (online=$online), fail=$fail"
exit $fail
