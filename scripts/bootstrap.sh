#!/usr/bin/env bash
#
# bootstrap.sh — restore the persistent dev env into a fresh RunPod container.
#
# Why this exists: on this pod only /workspace is a network volume. /root, /usr,
# /opt and /etc live on the container overlay and are WIPED on stop/start.
# This script rebuilds the ephemeral half from the persistent half.
#
# Run automatically at pod start (see README.md), or by hand any time:
#     bash /workspace/env/bootstrap.sh
#
# Idempotent: safe to run repeatedly, mid-session, or twice concurrently-ish.
# Never fatal — a failed step warns and the rest continues.
#
# Env overrides (testing):
#   ENV_HOME=/tmp/fakehome   pretend this is $HOME
#   BOOTSTRAP_TEST=1         skip system-level actions (apt, chsh, /etc, /opt)
#
set -uo pipefail

ENV_ROOT="${ENV_ROOT:-/workspace/env}"
H="${ENV_HOME:-${HOME:-/root}}"
TEST="${BOOTSTRAP_TEST:-0}"
PAY="$ENV_ROOT/home"
ZSH_BIN=/usr/bin/zsh          # deliberately NOT on /workspace: if the volume
                              # fails to mount, root must still have a shell.

log()  { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n'  "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }
sys()  { [ "$TEST" != 1 ]; }

log "bootstrap starting (home=$H, env=$ENV_ROOT)"

if [ ! -d "$ENV_ROOT" ]; then
  warn "$ENV_ROOT missing — is the network volume mounted? Nothing to restore."
  exit 1
fi

# --------------------------------------------------------------------- 1. packages
# The base image restores most of /usr on its own; only the delta our setup
# added needs reinstalling. Prefer the cached .debs (fast, works offline),
# fall back to apt.
APT_PKGS=(zsh zsh-common tmux git curl wget ca-certificates unzip fontconfig
          locales fzf zoxide eza ripgrep fd-find bat btop htop nvtop
          pkg-config python3-venv python3-pip)
missing=0
for c in zsh fzf zoxide eza rg fdfind batcat btop; do have "$c" || missing=1; done

if sys && [ "$missing" = 1 ]; then
  log "restoring apt packages…"
  if ls "$ENV_ROOT"/pkgs/*.deb >/dev/null 2>&1; then
    DEBIAN_FRONTEND=noninteractive dpkg -i "$ENV_ROOT"/pkgs/*.deb >/dev/null 2>&1 \
      || DEBIAN_FRONTEND=noninteractive apt-get install -f -y >/dev/null 2>&1 \
      || warn "dpkg restore incomplete"
  fi
  # still missing something? fall back to the network
  for c in zsh fzf zoxide eza rg fdfind batcat btop; do
    have "$c" || { log "apt fallback for missing tools…"
                   DEBIAN_FRONTEND=noninteractive apt-get update -y  >/dev/null 2>&1
                   DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
                     "${APT_PKGS[@]}" >/dev/null 2>&1 || warn "apt fallback failed"
                   break; }
  done
elif [ "$missing" = 0 ]; then
  log "apt packages already present — skipping"
fi

# fastfetch ships as a GitHub .deb, not an apt package
if sys && ! have fastfetch; then
  ff="$(ls "$ENV_ROOT"/pkgs/fastfetch-*.deb 2>/dev/null | head -1)"
  [ -n "$ff" ] && { log "installing fastfetch from cache…"
                    DEBIAN_FRONTEND=noninteractive apt-get install -y "$ff" >/dev/null 2>&1 \
                      || warn "fastfetch install failed"; }
fi

# --------------------------------------------------------------------- 2. link payload
# link SRC DEST — replace DEST with a symlink to SRC, backing up real files.
link() {
  local src="$1" dest="$2"
  [ -e "$src" ] || { warn "payload missing: $src"; return; }
  if [ -L "$dest" ]; then
    [ "$(readlink -f "$dest")" = "$(readlink -f "$src")" ] && return
    rm -f "$dest"
  elif [ -e "$dest" ]; then
    mv -f "$dest" "$dest.pre-bootstrap.$$" 2>/dev/null \
      && warn "backed up existing $dest -> $dest.pre-bootstrap.$$"
  fi
  mkdir -p "$(dirname "$dest")"
  ln -sfn "$src" "$dest"
}

log "linking home payload…"
# Everything at the top level of the payload gets linked into $H by name, so
# anything snapshot.sh adds later is picked up here with no code change.
if [ -d "$PAY" ]; then
  for p in "$PAY"/.[!.]* "$PAY"/*; do
    [ -e "$p" ] || continue
    link "$p" "$H/$(basename "$p")"
  done
fi

# ~/.local is a mixed bag (jupyter/applications are image-owned), so only the
# pieces we own get linked, rather than the whole directory.
mkdir -p "$H/.local/bin" "$H/.local/share"
for d in fonts claude; do
  [ -e "$ENV_ROOT/local/share/$d" ] && link "$ENV_ROOT/local/share/$d" "$H/.local/share/$d"
done

# claude launcher points at a version dir — resolve the newest one present
if [ -d "$H/.local/share/claude/versions" ]; then
  cv="$(ls -1 "$H/.local/share/claude/versions" 2>/dev/null | sort -V | tail -1)"
  [ -n "$cv" ] && ln -sfn "$H/.local/share/claude/versions/$cv" "$H/.local/bin/claude"
fi
# friendly names for Debian's renamed binaries
have fdfind  && ln -sfn "$(command -v fdfind)"  "$H/.local/bin/fd"
have batcat  && ln -sfn "$(command -v batcat)"  "$H/.local/bin/bat"

# --------------------------------------------------------------------- 3. /opt/nvim
if sys && [ -d "$ENV_ROOT/opt/nvim-linux-x86_64" ]; then
  link "$ENV_ROOT/opt/nvim-linux-x86_64" /opt/nvim-linux-x86_64
fi

# --------------------------------------------------------------------- 4. locale
if sys && ! locale -a 2>/dev/null | grep -qiE 'en_US\.utf-?8'; then
  log "regenerating en_US.UTF-8 locale…"
  grep -q '^en_US.UTF-8' /etc/locale.gen 2>/dev/null \
    || echo 'en_US.UTF-8 UTF-8' >> /etc/locale.gen
  locale-gen >/dev/null 2>&1 || warn "locale-gen failed"
fi

# --------------------------------------------------------------------- 5. fonts
if sys && have fc-cache && [ -d "$H/.local/share/fonts" ]; then
  fc-cache -f "$H/.local/share/fonts" >/dev/null 2>&1 || true
fi

# --------------------------------------------------------------------- 6. zsh as THE shell
# Order is load-bearing: only point root at zsh once zsh actually exists,
# otherwise a fresh container would have a login shell that isn't there.
if sys; then
  if [ -x "$ZSH_BIN" ]; then
    cur="$(getent passwd "$(id -un)" | cut -d: -f7)"
    if [ "$cur" != "$ZSH_BIN" ]; then
      log "setting login shell to zsh…"
      chsh -s "$ZSH_BIN" "$(id -un)" >/dev/null 2>&1 \
        || sed -i "s#^\(root:.*:\)[^:]*\$#\1$ZSH_BIN#" /etc/passwd 2>/dev/null \
        || warn "could not update login shell"
    fi
    # /etc/shells keeps chsh and some tools happy
    grep -qxF "$ZSH_BIN" /etc/shells 2>/dev/null || echo "$ZSH_BIN" >> /etc/shells

    # Catch-all for every entry point that hardcodes bash — RunPod's web
    # terminal (gotty), Jupyter terminals (terminado shell_command=/bin/bash)
    # and `docker exec bash` all read one of these.
    cat > /etc/profile.d/zzz-zsh-default.sh <<EOF
# hand interactive bash sessions over to zsh (set NO_AUTO_ZSH=1 to opt out)
if [ -z "\$ZSH_VERSION" ] && [ -x "$ZSH_BIN" ] && [ -t 1 ] && [ -z "\${NO_AUTO_ZSH:-}" ]; then
  export SHELL="$ZSH_BIN"; exec "$ZSH_BIN" -l
fi
EOF
    chmod 644 /etc/profile.d/zzz-zsh-default.sh
  else
    warn "zsh not installed — leaving login shell alone (this is the safe path)"
  fi
fi

# The same handoff for non-login interactive bash, which skips /etc/profile.d.
# .bashrc/.profile are part of the payload, so this is normally already there;
# re-assert it in case the image's copy won.
MARK="# >>> auto-exec zsh (setup.sh) >>>"
for rc in "$H/.bashrc" "$H/.profile"; do
  [ -e "$rc" ] || touch "$rc"
  grep -qF "$MARK" "$rc" 2>/dev/null || cat >> "$rc" <<EOF

$MARK
if [ -z "\$ZSH_VERSION" ] && [ -x "$ZSH_BIN" ] && [ -t 1 ] && [ -z "\${NO_AUTO_ZSH:-}" ]; then
  export SHELL="$ZSH_BIN"; exec "$ZSH_BIN" -l
fi
# <<< auto-exec zsh (setup.sh) <<<
EOF
done

# --------------------------------------------------------------------- 7. report
log "verifying…"
fail=0
export PATH="/opt/nvim-linux-x86_64/bin:$H/.local/bin:$PATH"
for c in zsh tmux starship nvim fastfetch eza fzf zoxide rg btop; do
  have "$c" || { warn "MISSING: $c"; fail=1; }
done
[ -e "$H/.zshrc" ]  || { warn "MISSING: ~/.zshrc"; fail=1; }
[ -e "$H/doots" ]   || { warn "MISSING: ~/doots"; fail=1; }
[ "$fail" = 0 ] && log "bootstrap OK — shell: $(getent passwd "$(id -un)" | cut -d: -f7)" \
                || warn "bootstrap finished with gaps (see above)"
exit 0
