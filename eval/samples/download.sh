#!/usr/bin/env bash
# Fetch a few short (<60s) Chinese tutorial videos into eval/samples/.
#
# Network is mainland China: prefer bilibili or generic HTTP sources over
# YouTube (usually blocked). Each entry below is "name|url"; yt-dlp handles
# both bilibili BV ids and plain http mp4 links. If a source fails, a hint is
# printed and the loop continues — the eval harness does not depend on any of
# these downloads succeeding.
set -uo pipefail

cd "$(dirname "$0")"
mkdir -p samples

# Add or replace these with your own short tutorial sources. Format: name|url
SOURCES=(
  # "bilibili_short_tutorial|https://www.bilibili.com/video/BV1xxxxxxxxxx"
  # "http_short_tutorial|https://example.com/sample.mp4"
)

if command -v yt-dlp >/dev/null 2>&1; then
  DL="yt-dlp"
else
  DL="python -m yt_dlp"
fi

for entry in "${SOURCES[@]}"; do
  name="${entry%%|*}"
  url="${entry#*|}"
  if [ -z "$url" ] || [ "$url" = "$entry" ]; then
    echo "hint: bad SOURCES entry '$entry' (expected name|url)"
    continue
  fi
  echo "downloading $name from $url"
  if $DL \
      --no-playlist \
      --format "best[ext=mp4]/best" \
      --merge-output-format mp4 \
      --output "samples/${name}.%(ext)s" \
      "$url" 2>&1; then
    echo "ok: $name"
  else
    echo "hint: failed to download $name ($url) — check network access to the source"
  fi
done

echo "done. samples: $(ls -1 samples/*.mp4 2>/dev/null | wc -l | tr -d ' ')"
