#!/bin/bash
# LOCAL HOST root only. Creates narrowly scoped aliases, never covers unrelated data.
set -euo pipefail
[[ $(id -u) -eq 0 ]] || { echo 'Run as root on the local Docker host' >&2; exit 77; }
exec 9>/run/lock/dsreset-prep-host.lock
flock 9
campaign=23cdfd5100ca3ded4557c748bc1623c5d0fd0e6a1b526190fde7d3c077af594f
[[ "$(docker inspect --format '{{.Id}}' "$campaign")" == "$campaign" ]]
actual=$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/home/ws"}}{{.Source}}{{end}}{{end}}' "$campaign")
[[ "$actual" == /ws/shkim ]] || { echo 'Campaign host mount changed; re-derive this exact mapping' >&2; exit 65; }
pid=$(docker inspect --format '{{.State.Pid}}' "$campaign")
[[ "$pid" =~ ^[1-9][0-9]*$ ]]
ensure_bind() {
    local src=$1 dst=$2 mode=$3
    [[ -d "$src" && ! -L "$src" && ! -L "$dst" ]]
    if mountpoint -q "$dst"; then
        [[ "$(stat -Lc '%d:%i' "$src")" == "$(stat -Lc '%d:%i' "$dst")" ]] || return 65
    else
        mkdir -p "$dst"
        [[ -z "$(find "$dst" -mindepth 1 -maxdepth 1 -print -quit)" ]] || return 65
        mount --bind "$src" "$dst"
    fi
    if [[ "$mode" == ro ]]; then mount -o remount,bind,ro "$dst"; fi
}
ensure_bind /ws/shkim /home/ws rw
a=/home/ws/ws/splatter4d/runs/rwv2-prep-assets
r=/home/compu/kaist/sunho/realworld
x=$r/processed/dsreset-20261113
d=/home/compu/kaist/sunho/data/real_world
mkdir -p "$a/canonical-realworld" /home/ws/data/realworld-processed/dsreset-20261113
ensure_bind "$a/canonical-realworld" "$r" rw
ensure_bind /home/ws/data/realworld-processed/dsreset-20261113 "$x" rw
ensure_bind /home/ws/data/real_world "$d" ro
while IFS= read -r entry; do
    grep -Fqx "$entry" /etc/fstab || printf '%s\n' "$entry" >>/etc/fstab
done <<'FSTAB'
/ws/shkim /home/ws none bind 0 0
/home/ws/ws/splatter4d/runs/rwv2-prep-assets/canonical-realworld /home/compu/kaist/sunho/realworld none bind,x-systemd.requires-mounts-for=/home/ws 0 0
/home/ws/data/realworld-processed/dsreset-20261113 /home/compu/kaist/sunho/realworld/processed/dsreset-20261113 none bind,x-systemd.requires-mounts-for=/home/compu/kaist/sunho/realworld 0 0
/home/ws/data/real_world /home/compu/kaist/sunho/data/real_world none bind,ro,x-systemd.requires-mounts-for=/home/ws 0 0
FSTAB
# The workspace container's mount propagation is not assumed. Use its current
# namespace/root; all sources below are its existing shared /home/ws bind.
nsenter --target "$pid" --mount --root --wd=/ -- /bin/bash -s <<'CONTAINER'
set -euo pipefail
ensure_bind() {
    local src=$1 dst=$2 mode=$3
    [[ -d "$src" && ! -L "$src" && ! -L "$dst" ]]
    if mountpoint -q "$dst"; then
        [[ "$(stat -Lc '%d:%i' "$src")" == "$(stat -Lc '%d:%i' "$dst")" ]] || return 65
    else
        mkdir -p "$dst"
        [[ -z "$(find "$dst" -mindepth 1 -maxdepth 1 -print -quit)" ]] || return 65
        mount --bind "$src" "$dst"
    fi
    if [[ "$mode" == ro ]]; then mount -o remount,bind,ro "$dst"; fi
}
ensure_bind /home/ws/ws/splatter4d/runs/rwv2-prep-assets/canonical-realworld /home/compu/kaist/sunho/realworld rw
ensure_bind /home/ws/data/realworld-processed/dsreset-20261113 /home/compu/kaist/sunho/realworld/processed/dsreset-20261113 rw
ensure_bind /home/ws/data/real_world /home/compu/kaist/sunho/data/real_world ro
CONTAINER
echo 'Canonical mounts verified in local host and current campaign namespace'
