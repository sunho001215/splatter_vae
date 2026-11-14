#!/bin/bash
# Coordinator runs this on the LOCAL Docker HOST. No operational invocation by implementer.
set -euo pipefail
bundle=${1:?immutable bundle directory name required}
[[ "$bundle" =~ ^local-(phaseA|calib-[a-z-]+|phaseB-[a-z-]+|records-[a-z-]+|measure-[a-z-]+)$ ]] || exit 64
mode=${2:---apply}
[[ "$mode" == --apply || "$mode" == --dry-run ]] || exit 64
pipeline=/home/ws/ws/realworld-ops/dsreset-20261113/pipeline
main=/home/ws/ws/splatter4d
canonical=/home/compu/kaist/sunho
campaign=23cdfd5100ca3ded4557c748bc1623c5d0fd0e6a1b526190fde7d3c077af594f
docker_bin=$(command -v docker)
[[ -x "$docker_bin" && -S /var/run/docker.sock ]]
[[ "$(docker inspect --format '{{.Id}}' "$campaign")" == "$campaign" ]]
[[ "$(docker inspect --format '{{.State.Running}}' "$campaign")" == true ]]
pid=$(docker inspect --format '{{.State.Pid}}' "$campaign")
[[ "$pid" =~ ^[1-9][0-9]*$ ]]
cg=$(awk -F: '$1=="0" {print $3}' "/proc/$pid/cgroup")
[[ "$cg" == /* && "$cg" != / && -f "/sys/fs/cgroup$cg/memory.current" ]]
image=$(cat "$pipeline/prep-image-id")
[[ "$image" =~ ^sha256:[a-f0-9]{64}$ ]]
[[ "$(docker image inspect --format '{{.Id}}' "$image")" == "$image" ]]
config=$pipeline/$bundle/dispatcher.json
[[ -f "$config" && -f "$main/experiments/queue.yaml" ]]
findmnt -M /home/ws >/dev/null
findmnt -M "$canonical/realworld" >/dev/null
findmnt -M "$canonical/realworld/processed/dsreset-20261113" >/dev/null
findmnt -M "$canonical/data/real_world" -no OPTIONS | tr ',' '\n' | grep -qx ro
name=s4d-dsreset-controller-$bundle
if [[ "$mode" == --dry-run ]]; then name=$name-dry; fi
fingerprint=$(sha256sum "$config" | cut -d' ' -f1)
if docker container inspect "$name" >/dev/null 2>&1; then
    [[ "$(docker inspect --format '{{index .Config.Labels "s4d.controller.config_sha256"}}' "$name")" == "$fingerprint" ]]
    [[ "$(docker inspect --format '{{.Config.Image}}' "$name")" == "$image" ]]
    [[ "$(docker inspect --format '{{.State.Running}}' "$name")" == true ]]
    exec docker attach --sig-proxy=false "$name"
fi
# Use accepted local MIGs for utility queries; no model runs in this controller.
devices=()
mapfile -t migs < <(docker exec --workdir "$main" "$campaign" /home/ws/ws/splatter4d/.venv/bin/python -c \
 'from ops.resources.gpu_guard import MIG_DEVICES; print("\n".join(MIG_DEVICES))')
[[ ${#migs[@]} -eq 8 ]]
for uuid in "${migs[@]}"; do
    [[ "$uuid" =~ ^MIG-[a-f0-9-]+$ ]] || exit 65
    devices+=(--device "nvidia.com/gpu=$uuid")
done
exec docker run --rm --name "$name" --read-only --pid="container:$campaign" --network none \
 --security-opt no-new-privileges --cap-drop ALL --memory 2g --memory-swap 2g --cpus 2 --pids-limit 512 \
 --tmpfs /tmp:rw,size=1g,mode=1777 \
 --label s4d.project=splatter4d --label s4d.owner=kaist-sunho --label s4d.role=rwprep-control \
 --label s4d.managed=false --label "s4d.controller.config_sha256=$fingerprint" \
 --mount type=bind,src=/home/ws,dst=/home/ws \
 --mount "type=bind,src=$canonical,dst=$canonical" \
 --mount "type=bind,src=/sys/fs/cgroup$cg,dst=/sys/fs/cgroup,readonly" \
 --mount "type=bind,src=$docker_bin,dst=/usr/local/bin/docker,readonly" \
 --mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock \
 --env PYTHONDONTWRITEBYTECODE=1 --env HOME=/tmp --env NVIDIA_DRIVER_CAPABILITIES=utility \
 "${devices[@]}" --workdir "$main" --entrypoint /bin/bash "$image" -c '
set -uo pipefail
main=$1; config=$2; mode=$3
while true; do
    flags=()
    [[ "$mode" == --dry-run ]] || flags=(--apply)
    # Scheduler holds registry.lock for its entire pass. The implementation also
    # takes stage2-driver.lock NONBLOCKING, so this outer order cannot deadlock
    # a local driver that holds stage2 and is waiting for registry. A busy pass
    # is skipped; every admitted tick serializes with BOTH local launch routes.
    flock -n -E 75 "$main/experiments/registry.lock" \
        /opt/s4d/.venv/bin/python -I -B "$main/ops/campaign/rwprep_mig.py" --config "$config" "${flags[@]}"
    rc=$?
    [[ "$rc" == 0 || "$rc" == 75 ]] || exit "$rc"
    [[ "$mode" == --dry-run ]] && exit "$rc"
    sleep 15
done
' controller "$main" "$config" "$mode"
