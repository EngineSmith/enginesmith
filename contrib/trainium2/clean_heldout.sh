# usage: clean_heldout.sh <es tree> <task> <round> <tag>
set -e
T=$1; TASK=$2; RD=$3; TAG=$4
R=$(ls -d $T/tasks/$TASK/runs/*/ | tail -1)
rm -rf /tmp/clean-$TAG && mkdir /tmp/clean-$TAG
git --git-dir=$R/engine.git archive refs/rounds/$RD | tar -x -C /tmp/clean-$TAG
cd /tmp/clean-$TAG && rm -rf target build
env -u NEURON_RT_VISIBLE_CORES ${BUILD_CORES:+NEURON_BUILD_CORES=$BUILD_CORES} ENGINESMITH_NEURON_CORE=${BUILD_BASE:-50} bash build.sh > ~/refbench/$TAG-clean-build.log 2>&1
cp target/engine ~/refbench/$TAG-clean-engine
sha256sum ~/refbench/$TAG-clean-engine
cd $T && .venv/bin/python ~/refbench/heldout.py tasks/$TASK/task.toml ~/refbench/$TAG-clean-engine ~/refbench/heldout-$TAG-clean ${REF_CORES:-44-45} ${ENG_CORES:-46-47}
