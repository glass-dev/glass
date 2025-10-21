#!/usr/bin/env bash

# Get the latest release tag and date
LATEST_TAG=$(gh release view --json tagName -q .tagName)
RELEASE_DATE=$(git log "$LATEST_TAG" -1 --format=%cI)

echo "Checking PRs merged after $RELEASE_DATE (tag: $LATEST_TAG)"
echo ""

# Get all merged PRs since the release date
gh pr list --state merged --json number,mergedAt,body,title | \
jq -r --arg date "$RELEASE_DATE" '
  .[] |
  select(.mergedAt > $date) |
  "\(.number)"
' | \
while read -r pr_number; do
  # Get PR details
  pr_data=$(gh pr view "$pr_number" --json body,title)
  title=$(echo "$pr_data" | jq -r .title)
  body=$(echo "$pr_data" | jq -r .body)

  # Extract the changelog section and remove HTML comments (including multiline)
  changelog_entries=$(echo "$body" | \
    sed -n '/## Changelog entry/,/^## /p' | \
    sed '/<!--/,/-->/d' | \
    sed '/^## Changelog entry/d' | \
    sed '/^## /d' | \
    sed '/^$/d' | \
    grep -v '^[[:space:]]*$')

  # Skip if it contains the template text (unchanged template)
  if echo "$changelog_entries" | grep -q "Added: Some new feature"; then
    continue
  fi

  # Only output if we found actual changelog entries
  if [ -n "$changelog_entries" ]; then
    echo "=== PR #$pr_number: $title ==="
    echo "${changelog_entries//^/  }"
    echo ""
  fi
done
