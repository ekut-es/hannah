# !/usr/bin/env bash
# this script will

CONFIG_1="ofa_nas_group_test_long"
CONFIG_2="ofa_nas_group_test_long_no_grouping"
PATH_TO_HANNAH="/local/raugustm/hannah/hannah"
PATH_TO_ROOT="/local/raugustm/hannah/"

BRANCH_TAG="quant_branch"

CONTROL_LOG_FILE="model.log"

red="\e[0;91m"
blue="\e[0;94m"
yellow="\e[0;33m"
expand_bg="\e[K"
blue_bg="\e[0;104m${expand_bg}"
red_bg="\e[0;101m${expand_bg}"
green_bg="\e[0;102m${expand_bg}"
green="\e[0;92m"
white="\e[0;97m"
bold="\e[1m"
uline="\e[4m"
reset="\e[0m"



function switch_branch {
    git checkout "$1"
}

function switch_master {
    switch_branch "master"
}

function switch_work_branch {
    switch_branch "wip_mr_ofa_grouping_quant"
}


function do_run_normal {

    do_run "ofa_group_test" "${BRANCH_TAG}_group_yes" "${BRANCH_TAG}_group_no"
}
function do_run_quant {

    do_run "ofa_group_quant" "${BRANCH_TAG}_quant_group_yes" "${BRANCH_TAG}_quant_group_no" "$1"
}

function do_control_group {
    do_run "ofa_quant" "${BRANCH_TAG}_quant_group_yes_control" "${BRANCH_TAG}_quant_group_no_control" "$1"
}

function do_larger_group_set {
    # TODO
    do_run "ofa_group_quantv2" "${BRANCH_TAG}_quant_group_yes" "${BRANCH_TAG}_quant_group_no" "$1"
}

function do_run {

    let model
    model=$1
    # used for master, cause it has no quant grouping yet
    skip_grouping="${4:-false}"

    if [[ "$skip_grouping" != "true" ]]; then
        echo -e "$green"
        echo "Doing hannah config ${CONFIG_1} with grouping" | tee -a $CONTROL_LOG_FILE
        echo -e "$yellow"
        cat "${PATH_TO_HANNAH}/conf/nas/${CONFIG_1}.yaml" |  sed 's/^/    /' | tee -a $CONTROL_LOG_FILE
        let experiment_id=$2
        experiment_id=$2
        echo -e "$green"
        echo "using id '$experiment_id' with model '$model'" | tee -a $CONTROL_LOG_FILE
        #hannah-train --config-name nas_ofa model="$model" experiment_id="$experiment_id"  nas="${CONFIG_1}" module.num_workers=8
        echo -e $reset
        echo "hannah-train --config-name nas_ofa model="$model" experiment_id="$experiment_id"  nas="${CONFIG_1}" module.num_workers=8"
            hannah-train --config-name nas_ofa model="$model" experiment_id="$experiment_id"  nas="${CONFIG_1}" module.num_workers=8
    fi

    # no grouping
    echo -e "$green"
    echo "Doing hannah config ${CONFIG_2} without grouping" | tee -a $CONTROL_LOG_FILE
    echo -e "$yellow"
    cat "${PATH_TO_HANNAH}/conf/nas/${CONFIG_2}.yaml" |  sed 's/^/    /' | tee -a $CONTROL_LOG_FILE
    echo -e "$green"
    experiment_id=$3
    echo "using id '$experiment_id' with model '$model'" | tee -a $CONTROL_LOG_FILE
    echo -e $reset

    echo "hannah-train --config-name nas_ofa model="$model" experiment_id="$experiment_id"  nas="${CONFIG_2}" module.num_workers=8"
    hannah-train --config-name nas_ofa model="$model" experiment_id="$experiment_id"  nas="${CONFIG_2}" module.num_workers=8
}

#set -e
echo "Running now train batch:"
start=`date +%s`
echo $(date) | tee $CONTROL_LOG_FILE
echo "branch: $BRANCH_TAG"
do_run_normal
echo "1:Done"
echo "Running now quantazied version:"

skip_grouping="false"
if [[ "$BRANCH_TAG" == "master" ]]; then
    skip_grouping="true"
    echo -e "$green"
    echo "Skip grouping for $BRANCH_TAG"
    #hannah-train --config-name nas_ofa model="$model" experiment_id="$experiment_id"  nas="${CONFIG_1}" module.num_workers=8
    echo -e $reset
fi
# skip master group quant
do_run_quant $skip_grouping
echo "2:Done"
echo "Running now Control Group with model:ofa_quant"
do_control_group "true"
echo "Train batch complete"
echo $(date) | tee -a $CONTROL_LOG_FILE
ls -la $PATH_TO_ROOT/trained_models
end=`date +%s`
echo "Elapsed Time: $(($end-$start)) seconds; $((($end-$start)/60)) in Minutes" | tee -a $CONTROL_LOG_FILE
