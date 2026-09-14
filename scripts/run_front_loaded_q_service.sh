#!/usr/bin/env bash
set -Eeuo pipefail

run_name=${1:?run name is required}
shift
if [[ ! ${run_name} =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "invalid run name: ${run_name}" >&2
    exit 2
fi
if [[ $# -eq 0 ]]; then
    echo "worker command is required" >&2
    exit 2
fi

# User services do not inherit the interactive shell's virtual-environment PATH.
# Resolve the conventional Python command before recording or executing it.
project_python=${QUANTDATA_PROJECT_PYTHON:-/home/hans/vqdata/bin/python}
if [[ $1 == python ]]; then
    if [[ ! -x ${project_python} ]]; then
        echo "project Python is not executable: ${project_python}" >&2
        exit 2
    fi
    shift
    set -- "${project_python}" "$@"
fi

artifact_root=/home/hans/quantdata_data/training/flex-fixed-grid/four-week-no-am-v1/q-existence/front-loaded-pfree-w2-v1
log_path=${artifact_root}/logs/${run_name}.log
status_path=${artifact_root}/status/${run_name}.json
started_at=$(date --iso-8601=seconds)
worker_pid=

mkdir -p "${artifact_root}/logs" "${artifact_root}/status"
cd /home/hans/python/quant_data

export TMPDIR=/tmp
export PYTHONDONTWRITEBYTECODE=1
export QUANTDATA_ROOT=/home/hans/quantdata_data
export QUANTDATA_RAW_DATA=/mnt/nas/quantdata
export QUANTDATA_PROCESSING_DATA=/home/hans/quantdata_data/processing
export QUANTDATA_TRAINING_DATA=/home/hans/quantdata_data/training
export QUANTDATA_EXPERIMENTS=/home/hans/quantdata_data/experiments
export QUANTDATA_TEMP_DATA=/home/hans/quantdata_data/tmp/quantdata

write_status() {
    local state=$1
    local exit_code=${2:-null}
    local updated_at
    local temporary
    updated_at=$(date --iso-8601=seconds)
    temporary=${status_path}.tmp.$$
    printf '{"run_name":"%s","state":"%s","wrapper_pid":%d,"worker_pid":%s,"started_at":"%s","updated_at":"%s","exit_code":%s,"log":"%s"}\n' \
        "${run_name}" \
        "${state}" \
        "$$" \
        "${worker_pid:-null}" \
        "${started_at}" \
        "${updated_at}" \
        "${exit_code}" \
        "${log_path}" > "${temporary}"
    mv "${temporary}" "${status_path}"
}

handle_signal() {
    write_status interrupted null
    if [[ -n ${worker_pid} ]]; then
        kill -TERM "${worker_pid}" 2>/dev/null || true
    fi
}
trap handle_signal TERM INT

printf '\n[%s] starting %s\n' "${started_at}" "${run_name}" >> "${log_path}"
printf '[%s] command:' "${started_at}" >> "${log_path}"
printf ' %q' "$@" >> "${log_path}"
printf '\n' >> "${log_path}"

"$@" >> "${log_path}" 2>&1 &
worker_pid=$!
write_status running null

while kill -0 "${worker_pid}" 2>/dev/null; do
    write_status running null
    sleep 20
done

set +e
wait "${worker_pid}"
exit_code=$?
set -e
if [[ ${exit_code} -eq 0 ]]; then
    final_state=complete
else
    final_state=failed
fi
write_status "${final_state}" "${exit_code}"
printf '[%s] finished %s with exit code %d\n' \
    "$(date --iso-8601=seconds)" "${run_name}" "${exit_code}" >> "${log_path}"
exit "${exit_code}"
