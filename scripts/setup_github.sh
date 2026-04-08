#!/usr/bin/env bash
# ============================================================
# Helix — GitHub repository setup script
#
# This script initialises a GitHub remote and configures the
# required secrets so CI/CD can build and push Docker images.
#
# Prerequisites:
#   1. GitHub CLI installed: https://cli.github.com/
#   2. Logged in: gh auth login
#   3. Docker Hub account at hub.docker.com
#
# Usage:
#   chmod +x scripts/setup_github.sh
#   ./scripts/setup_github.sh --repo YOUR_GITHUB_USERNAME/helix
# ============================================================
set -euo pipefail

REPO=""
VISIBILITY="private"

usage() {
  echo "Usage: $0 --repo <owner/repo> [--public]"
  echo "  --repo    GitHub repo in owner/repo format (required)"
  echo "  --public  Create as public repo (default: private)"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --repo)   REPO="$2"; shift 2 ;;
    --public) VISIBILITY="public"; shift ;;
    *)        usage ;;
  esac
done

[[ -z "$REPO" ]] && usage

echo "==> Checking GitHub CLI..."
gh --version

echo "==> Checking auth status..."
gh auth status

# ── Create repo if it doesn't exist ───────────────────────────────────────────
echo ""
echo "==> Creating GitHub repository: $REPO ($VISIBILITY)..."
gh repo create "$REPO" \
  --"$VISIBILITY" \
  --description "Helix — end-to-end ML platform for financial market anomaly detection" \
  --clone=false \
  2>/dev/null || echo "   (repo may already exist — continuing)"

# ── Set remote ────────────────────────────────────────────────────────────────
echo ""
echo "==> Configuring git remote..."
if git remote get-url origin &>/dev/null; then
  echo "   Remote 'origin' already set to: $(git remote get-url origin)"
  read -rp "   Overwrite? [y/N] " ans
  if [[ "${ans,,}" == "y" ]]; then
    git remote set-url origin "https://github.com/$REPO.git"
  fi
else
  git remote add origin "https://github.com/$REPO.git"
fi
echo "   Remote: $(git remote get-url origin)"

# ── Docker Hub secrets ────────────────────────────────────────────────────────
echo ""
echo "==> Setting Docker Hub secrets..."
echo "   You need a Docker Hub account and an Access Token."
echo "   Create a token at: https://hub.docker.com/settings/security"
echo ""

read -rp "   Docker Hub username: " DOCKERHUB_USERNAME
read -rsp "  Docker Hub token (input hidden): " DOCKERHUB_TOKEN
echo ""

gh secret set DOCKERHUB_USERNAME --body "$DOCKERHUB_USERNAME" --repo "$REPO"
gh secret set DOCKERHUB_TOKEN    --body "$DOCKERHUB_TOKEN"    --repo "$REPO"
echo "   Secrets DOCKERHUB_USERNAME and DOCKERHUB_TOKEN set."

# ── Optional: production server SSH secret ────────────────────────────────────
echo ""
read -rp "==> Configure deployment SSH key? (for CD pipeline) [y/N] " setup_ssh
if [[ "${setup_ssh,,}" == "y" ]]; then
  read -rp "   Path to SSH private key [~/.ssh/id_rsa]: " KEY_PATH
  KEY_PATH="${KEY_PATH:-$HOME/.ssh/id_rsa}"
  read -rp "   Production server host (e.g. user@1.2.3.4): " DEPLOY_HOST

  gh secret set DEPLOY_SSH_KEY  --body "$(cat "$KEY_PATH")" --repo "$REPO"
  gh secret set DEPLOY_HOST     --body "$DEPLOY_HOST"       --repo "$REPO"
  echo "   SSH secrets set."
fi

# ── detect-secrets baseline ───────────────────────────────────────────────────
echo ""
echo "==> Initialising detect-secrets baseline..."
if command -v detect-secrets &>/dev/null; then
  detect-secrets scan > .secrets.baseline
  echo "   .secrets.baseline created"
else
  echo "   detect-secrets not installed — run: pip install detect-secrets"
  echo "   Then run: detect-secrets scan > .secrets.baseline"
fi

# ── Initial push ──────────────────────────────────────────────────────────────
echo ""
read -rp "==> Push local code to GitHub now? [y/N] " do_push
if [[ "${do_push,,}" == "y" ]]; then
  git add .
  git status --short
  read -rp "   Commit message [Initial commit]: " MSG
  MSG="${MSG:-Initial commit}"
  git commit -m "$MSG" --allow-empty
  git push -u origin main
  echo "   Pushed to https://github.com/$REPO"
fi

echo ""
echo "============================================================"
echo "GitHub setup complete!"
echo ""
echo "Next steps:"
echo "  1. Visit https://github.com/$REPO/actions to see CI runs"
echo "  2. Create a branch and open a PR to trigger the full pipeline"
echo "  3. After merging to main, CD will build and push Docker images"
echo ""
echo "Required secrets configured:"
echo "  DOCKERHUB_USERNAME  ✓"
echo "  DOCKERHUB_TOKEN     ✓"
echo "============================================================"
