#!/usr/bin/env bash
set -euo pipefail

# Sync generated images from Ubuntu server to local ~/Pictures, skipping existing files.
# Usage:
#   REMOTE_HOST=server-ip REMOTE_USER=master-benjamin REMOTE_DIR=/home/master-benjamin/Pictures ./scripts/sync_pictures.sh
# Defaults:
REMOTE_HOST=${REMOTE_HOST:-"192.168.0.1"}
REMOTE_USER=${REMOTE_USER:-"master-benjamin"}
REMOTE_DIR=${REMOTE_DIR:-"/home/${REMOTE_USER}/Pictures"}
LOCAL_DIR=${LOCAL_DIR:-"$HOME/Pictures"}

mkdir -p "$LOCAL_DIR"
echo "Syncing from ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/ -> ${LOCAL_DIR}/ (skip existing)"
rsync -avh --ignore-existing --progress -e ssh "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/" "${LOCAL_DIR}/"
echo "Done."

