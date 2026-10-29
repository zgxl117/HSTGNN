#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'HELP'
HSTGNN + SAC-MOEA 启动器

  bash run.sh ui          终端交互界面（默认）
  bash run.sh smoke       完整闭环冒烟测试
  bash run.sh run         完整预算闭环
  bash run.sh reuse       使用已有预测进行加固
  bash run.sh check       检查环境依赖
  bash run.sh datasets    列出数据 YAML
  bash run.sh test        运行回归测试

pipeline 参数可直接跟在 smoke/run/reuse 后：
  bash run.sh smoke --dataset /path/to/dataset.yaml
  bash run.sh reuse --dataset /path/to/dataset.yaml \
    --train-config configs/task_train.yaml --predictions /path/to/results.csv

环境变量（均为单个路径或值，不作为 Shell 命令执行）：
  HSTGNN_PYTHON        Python 可执行文件路径
  HSTGNN_DATASET       数据 YAML，默认 RISC-V test2
  HSTGNN_PROFILE       加固 profile（可选）
  HSTGNN_RUN_NAME      新运行名称（可选，不覆盖旧目录）
  HSTGNN_TRAIN_CONFIG  已有训练配置，配合 reuse
  HSTGNN_PREDICTIONS   已有预测 CSV，配合 reuse
  HSTGNN_SMOKE=1       reuse 采用小优化预算
  HSTGNN_DRY_RUN=1     仅打印命令，不执行

Python 优先级：显式指定 > 已激活的非 base Conda > 工程 .venv > 本机 ckt > PATH。
脚本会自动进入工程根目录；相对配置路径以工程根目录解析。
HELP
}

action="${1:-ui}"
if [[ $# -gt 0 ]]; then shift; fi
case "$action" in
  help|-h|--help) usage; exit 0 ;;
  ui|smoke|run|reuse|check|datasets|test) ;;
  *) printf '未知命令：%s\n' "$action" >&2; usage >&2; exit 2 ;;
esac

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$project_dir"

if [[ -n "${HSTGNN_PYTHON:-}" ]]; then
  python_bin="$HSTGNN_PYTHON"
elif [[ -n "${CONDA_PREFIX:-}" && "${CONDA_DEFAULT_ENV:-base}" != base && -x "$CONDA_PREFIX/bin/python" ]]; then
  python_bin="$CONDA_PREFIX/bin/python"
elif [[ -x "$project_dir/.venv/bin/python" ]]; then
  python_bin="$project_dir/.venv/bin/python"
elif [[ -x "$HOME/anaconda3/envs/ckt/bin/python" ]]; then
  python_bin="$HOME/anaconda3/envs/ckt/bin/python"
else
  python_bin="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$python_bin" ]] || ! command -v -- "$python_bin" >/dev/null 2>&1; then
  printf '找不到 Python，请设置 HSTGNN_PYTHON=/path/to/python。\n' >&2
  exit 127
fi

case "$action" in
  ui) arguments=(terminal_ui.py "$@") ;;
  check) arguments=(terminal_ui.py --diagnose "$@") ;;
  datasets) arguments=(terminal_ui.py --list-datasets "$@") ;;
  test) arguments=(-m pytest tests -q "$@") ;;
  smoke|run|reuse)
    dataset="${HSTGNN_DATASET:-../AI4FaultSim/fault-sim/FsimNNs/data/data/riscy_test/test2/dataset.yaml}"
    arguments=(pipeline.py --dataset "$dataset")
    if [[ "$action" == smoke || ( "$action" == reuse && "${HSTGNN_SMOKE:-0}" == 1 ) ]]; then
      arguments+=(--smoke)
    fi
    if [[ -n "${HSTGNN_PROFILE:-}" ]]; then arguments+=(--profile "$HSTGNN_PROFILE"); fi
    if [[ -n "${HSTGNN_RUN_NAME:-}" ]]; then arguments+=(--name "$HSTGNN_RUN_NAME"); fi
    if [[ -n "${HSTGNN_TRAIN_CONFIG:-}" ]]; then arguments+=(--train-config "$HSTGNN_TRAIN_CONFIG"); fi
    if [[ -n "${HSTGNN_PREDICTIONS:-}" ]]; then arguments+=(--predictions "$HSTGNN_PREDICTIONS"); fi
    if [[ "$action" == reuse ]]; then
      has_config=0
      has_predictions=0
      for argument in "${arguments[@]}" "$@"; do
        case "$argument" in
          --train-config|--train-config=*) has_config=1 ;;
          --predictions|--predictions=*) has_predictions=1 ;;
        esac
      done
      if [[ "$has_config" != 1 || "$has_predictions" != 1 ]]; then
        printf 'reuse 需要 --train-config 和 --predictions，或对应的 HSTGNN 环境变量。\n' >&2
        exit 2
      fi
    fi
    arguments+=("$@")
    ;;
esac

if [[ "${HSTGNN_DRY_RUN:-0}" == 1 ]]; then
  printf 'cwd=%q\n' "$project_dir"
  printf '%q ' "$python_bin" "${arguments[@]}"
  printf '\n'
  exit 0
fi
exec "$python_bin" "${arguments[@]}"
