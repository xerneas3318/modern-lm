#!/usr/bin/env bash
#
# snapshot.sh — push the live environment in $HOME onto the persistent volume.
#
# First run migrates; later runs refresh. Run it after installing new tools, and
# before stopping the pod if you want the very latest Claude/shell state kept.
#     bash /workspace/env/snapshot.sh
#
# Two classes of state, handled differently on purpose:
#
#   INERT  — dotfiles, plugins, toolchains. Nothing holds them open, so they are
#            MOVED to the volume and symlinked back. Zero drift from here on:
#            edits to ~/.zshrc are edits to the volume copy.
#
#   LIVE   — the Claude Code install and its state dir. A running process is
#            executing out of these, so they are COPIED, never moved. bootstrap.sh
#            converts them to symlinks at next boot, when nothing is using them.
#
# Credentials: ~/.claude/.credentials.json is EXCLUDED by default — it is an auth
# token and the volume outlives the pod and can be mounted by other pods. Opt in
# with PERSIST_CREDENTIALS=1 if you would rather not re-login after a restart.
#
set -uo pipefail

ENV_ROOT="${ENV_ROOT:-/workspace/env}"
H="${ENV_HOME:-${HOME:-/root}}"
PAY="$ENV_ROOT/home"
PERSIST_CREDENTIALS="${PERSIST_CREDENTIALS:-0}"

log()  { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n'  "$*"; }
skip() { printf '\033[1;34m -\033[0m %s\n'   "$*"; }

mkdir -p "$PAY" "$ENV_ROOT"/{pkgs,log,opt,local/share} || {
  warn "cannot write $ENV_ROOT — volume mounted?"; exit 1; }

# ---------------------------------------------------------------- inert: move + link
INERT=(.zshrc .zshenv .bashrc .profile .gitconfig .tmux.conf .tmux .zsh .config
       .nvm .cargo .rustup doots)

log "capturing inert state (move + symlink)…"
for name in "${INERT[@]}"; do
  src="$H/$name" dst="$PAY/$name"
  if [ -L "$src" ]; then
    # already pointing at the payload: nothing to do
    [ "$(readlink -f "$src")" = "$(readlink -f "$dst")" ] && { skip "$name (already linked)"; continue; }
    warn "$name is a symlink elsewhere ($(readlink "$src")) — leaving alone"; continue
  fi
  [ -e "$src" ] || { skip "$name (absent)"; continue; }
  if [ -e "$dst" ]; then
    # payload already has it and $HOME has a real copy: $HOME wins, keep a backup
    mv -f "$dst" "$dst.replaced.$$" && warn "$name: replaced payload copy (old -> $(basename "$dst").replaced.$$)"
  fi
  if mv "$src" "$dst" 2>/dev/null; then
    ln -sfn "$dst" "$src"; log "$name -> volume"
  else
    warn "$name: move failed, leaving in place"
  fi
done

# ---------------------------------------------------------------- fonts (inert, nested)
if [ -d "$H/.local/share/fonts" ] && [ ! -L "$H/.local/share/fonts" ]; then
  if mv "$H/.local/share/fonts" "$ENV_ROOT/local/share/fonts" 2>/dev/null; then
    ln -sfn "$ENV_ROOT/local/share/fonts" "$H/.local/share/fonts"; log "fonts -> volume"
  else warn "fonts: move failed"; fi
else
  skip "fonts (already linked or absent)"
fi

# ---------------------------------------------------------------- live: copy only
copy_live() { # src dest [rsync extra args…]
  local src="$1" dst="$2"; shift 2
  [ -e "$src" ] || { skip "$(basename "$src") (absent)"; return; }
  if [ -L "$src" ] && [ "$(readlink -f "$src")" = "$(readlink -f "$dst")" ]; then
    skip "$(basename "$src") (already linked — live-persistent)"; return
  fi
  mkdir -p "$(dirname "$dst")"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "$@" "$src"/ "$dst"/ 2>/dev/null && log "$(basename "$src") copied -> volume" \
      || warn "$(basename "$src"): rsync had errors (files in flight are normal)"
  else
    cp -a "$src/." "$dst/" 2>/dev/null && log "$(basename "$src") copied -> volume" || warn "copy failed"
  fi
}

log "capturing live state (copy — in use by the running session)…"
CRED_ARGS=()
if [ "$PERSIST_CREDENTIALS" != 1 ]; then
  CRED_ARGS=(--exclude '.credentials.json')
fi
copy_live "$H/.claude" "$PAY/.claude" \
  "${CRED_ARGS[@]}" --exclude 'shell-snapshots/' --exclude 'cache/' --exclude 'sessions/'
copy_live "$H/.local/share/claude" "$ENV_ROOT/local/share/claude"
[ -f "$H/.claude.json" ] && { cp -a "$H/.claude.json" "$PAY/.claude.json" && log ".claude.json copied"; }

if [ "$PERSIST_CREDENTIALS" != 1 ]; then
  warn "~/.claude/.credentials.json NOT persisted (auth token). Re-run with"
  warn "PERSIST_CREDENTIALS=1 to keep it, or just run 'claude' and log in again."
fi

# ---------------------------------------------------------------- /opt/nvim
if [ -d /opt/nvim-linux-x86_64 ] && [ ! -L /opt/nvim-linux-x86_64 ]; then
  if mv /opt/nvim-linux-x86_64 "$ENV_ROOT/opt/nvim-linux-x86_64" 2>/dev/null; then
    ln -sfn "$ENV_ROOT/opt/nvim-linux-x86_64" /opt/nvim-linux-x86_64; log "nvim -> volume"
  else warn "nvim: move failed"; fi
else
  skip "nvim (already linked or absent)"
fi

# ---------------------------------------------------------------- deb cache
if [ ! -f "$ENV_ROOT/pkgs/.cached" ]; then
  warn "no .deb cache yet — run: bash $ENV_ROOT/cache-pkgs.sh"
fi

log "snapshot complete. Volume usage:"
du -sh "$ENV_ROOT" 2>/dev/null
