#!/usr/bin/env bash
# Sync the repo to the RunPod pod. Run from local machine.
# Usage:
#   POD_IP=213.173.107.18 POD_PORT=31243 bash scripts/deploy_to_droplet.sh
set -euo pipefail

POD_IP="${POD_IP:?POD_IP env var required}"
POD_PORT="${POD_PORT:-22}"
POD_USER="${POD_USER:-root}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/llm-final-project}"

rsync -rltDvz --progress \
    --no-owner --no-group --no-perms --chmod=ugo=rwX \
    -e "ssh -p ${POD_PORT} -i ${SSH_KEY} -o StrictHostKeyChecking=accept-new" \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude 'outputs/checkpoints' \
    --exclude 'data/raw/meb_tarih/*.pdf' \
    --exclude '.git' \
    --exclude 'wandb' \
    --exclude '.cache' \
    --exclude '.DS_Store' \
    ./ "${POD_USER}@${POD_IP}:${REMOTE_DIR}/"

echo "[ok] synced to ${POD_USER}@${POD_IP}:${REMOTE_DIR}"
echo "Next:"
echo "  ssh -p ${POD_PORT} -i ${SSH_KEY} ${POD_USER}@${POD_IP}"
echo "  cd ${REMOTE_DIR} && bash scripts/setup_gpu.sh"
