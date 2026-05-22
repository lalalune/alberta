#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/squash_to_single_commit.sh [options]

Dry-run by default. Prints the exact rewrite plan without changing git history.

Options:
  --execute                 Actually rewrite history.
  --message TEXT            Commit message for the squashed commit.
                            Default: "chore: squash repository history"
  --base REV                Preserve history through REV and squash commits after REV.
                            Without --base, rewrites the current branch to one orphan commit.
  --include-uncommitted     Include current working tree/index changes in the squashed commit.
                            Without this flag, --execute requires a clean worktree.
  --backup-branch NAME      Backup branch name. Default: backup/pre-squash-<branch>-<timestamp>
  -h, --help                Show this help.

Examples:
  scripts/squash_to_single_commit.sh
  scripts/squash_to_single_commit.sh --execute --message "chore: initial import"
  scripts/squash_to_single_commit.sh --base origin/main
USAGE
}

mode="dry-run"
message="chore: squash repository history"
base_rev=""
include_uncommitted="false"
backup_branch=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute)
      mode="execute"
      shift
      ;;
    --message)
      if [[ $# -lt 2 ]]; then
        echo "error: --message requires a value" >&2
        exit 2
      fi
      message="$2"
      shift 2
      ;;
    --base)
      if [[ $# -lt 2 ]]; then
        echo "error: --base requires a revision" >&2
        exit 2
      fi
      base_rev="$2"
      shift 2
      ;;
    --include-uncommitted)
      include_uncommitted="true"
      shift
      ;;
    --backup-branch)
      if [[ $# -lt 2 ]]; then
        echo "error: --backup-branch requires a name" >&2
        exit 2
      fi
      backup_branch="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$branch" == "HEAD" ]]; then
  echo "error: detached HEAD is not supported" >&2
  exit 1
fi

head_commit="$(git rev-parse HEAD)"
timestamp="$(date +%Y%m%d-%H%M%S)"
safe_branch="${branch//\//-}"
if [[ -z "$backup_branch" ]]; then
  backup_branch="backup/pre-squash-${safe_branch}-${timestamp}"
fi

upstream="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
status_short="$(git status --short)"
commit_count="$(git rev-list --count HEAD)"
root_commits="$(git rev-list --max-parents=0 --all | wc -l | tr -d ' ')"

if [[ -n "$base_rev" ]]; then
  git rev-parse --verify "${base_rev}^{commit}" >/dev/null
  base_commit="$(git rev-parse "${base_rev}^{commit}")"
  if ! git merge-base --is-ancestor "$base_commit" HEAD; then
    echo "error: base revision is not an ancestor of HEAD: $base_rev" >&2
    exit 1
  fi
  rewrite_kind="base-preserving soft reset"
  commits_to_squash="$(git rev-list --count "${base_commit}..HEAD")"
else
  base_commit=""
  rewrite_kind="whole-branch orphan rewrite"
  commits_to_squash="$commit_count"
fi

if [[ "$mode" == "execute" && "$include_uncommitted" != "true" && -n "$status_short" ]]; then
  echo "error: worktree has uncommitted changes. Re-run with --include-uncommitted to include them." >&2
  git status --short >&2
  exit 1
fi

echo "Repository:      $repo_root"
echo "Branch:          $branch"
echo "Upstream:        ${upstream:-<none>}"
echo "HEAD:            $head_commit"
echo "Commits on HEAD: $commit_count"
echo "Root commits:    $root_commits"
echo "Rewrite kind:    $rewrite_kind"
if [[ -n "$base_commit" ]]; then
  echo "Base:            $base_rev ($base_commit)"
fi
echo "To squash:       $commits_to_squash commits"
echo "Backup branch:   $backup_branch"
echo "Commit message:  $message"
echo "Mode:            $mode"
echo

if [[ -n "$status_short" ]]; then
  echo "Current worktree changes:"
  git status --short
  echo
else
  echo "Current worktree changes: <clean>"
  echo
fi

echo "Planned commands:"
echo "  git branch \"$backup_branch\" \"$head_commit\""
if [[ -n "$base_commit" ]]; then
  echo "  git reset --soft \"$base_commit\""
  echo "  git add -A"
  echo "  git commit -m \"$message\""
else
  temp_branch="squash-single-${safe_branch}-${timestamp}"
  echo "  git switch --orphan \"$temp_branch\""
  echo "  git add -A"
  echo "  git commit -m \"$message\""
  echo "  git branch -M \"$temp_branch\" \"$branch\""
fi
if [[ -n "$upstream" ]]; then
  echo "  # Push later with: git push --force-with-lease ${upstream%%/*} \"$branch\""
else
  echo "  # Push later with: git push --force-with-lease <remote> \"$branch\""
fi
echo

if [[ "$mode" != "execute" ]]; then
  echo "Dry run only. No history was rewritten."
  exit 0
fi

git branch "$backup_branch" "$head_commit"

if [[ -n "$base_commit" ]]; then
  git reset --soft "$base_commit"
  git add -A
  git commit -m "$message"
else
  temp_branch="squash-single-${safe_branch}-${timestamp}"
  git switch --orphan "$temp_branch"
  git add -A
  git commit -m "$message"
  git branch -M "$temp_branch" "$branch"
fi

echo
echo "Squash complete."
echo "Backup branch: $backup_branch"
echo "New HEAD:      $(git rev-parse HEAD)"
