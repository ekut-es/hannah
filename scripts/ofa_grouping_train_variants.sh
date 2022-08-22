# !/usr/bin/env bash
# this script will

CONFIG_1="ofa_nas_group_test_long"
CONFIG_2="ofa_nas_group_test_long_no_grouping"
PATH_TO_HANNAH="/local/raugustm/hannah/hannah"



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

    do_run "ofa_group_test" "nas_ofa_group_test_g_long" "nas_ofa_group_test_no_grouping_long"
    # let model
    # model="ofa_group_test"

    # echo "Doing hannah config ${CONFIG_1} with grouping"
    # cat "${PATH_TO_HANNAH}/conf/nas/${CONFIG_1}.yml"
    # let experiment_id="nas_ofa_group_test_g_long"
    # echo "$experiment_id with model $model"
    # hannah-train --config-name nas_ofa model="$model" experiment_id="$experiment_id"  nas="${CONFIG_1}" module.num_workers=8

    # # no grouping
    # echo "Doing hannah config ${CONFIG_2} without grouping"
    # cat "${PATH_TO_HANNAH}/conf/nas/${CONFIG_2}.yml"
    # let experiment_id="nas_ofa_group_test_no_grouping_long"
    # echo "$experiment_id with model $model"
    # hannah-train --config-name nas_ofa model="$model" experiment_id="$experiment_id"  nas="${CONFIG_2}" module.num_workers=8
}
function do_run_quant {

    do_run "ofa_group_quant" "quant_nas_ofa_group_test_g_long" "quant_nas_ofa_group_test_no_grouping_long"
    # let model
    # model="ofa_group_quant"

    # echo "Doing hannah config ${CONFIG_1} with grouping"
    # cat "${PATH_TO_HANNAH}/conf/nas/${CONFIG_1}.yml"
    # let experiment_id="quant_nas_ofa_group_test_g_long"
    # echo "$experiment_id with model $model"
    # hannah-train --config-name nas_ofa model="$model" experiment_id="$experiment_id"  nas="${CONFIG_1}" module.num_workers=8

    # # no grouping
    # echo "Doing hannah config ${CONFIG_2} without grouping"
    # cat "${PATH_TO_HANNAH}/conf/nas/${CONFIG_2}.yml"
    # let experiment_id="quant_nas_ofa_group_test_no_grouping_long"
    # echo "$experiment_id with model $model"
    # hannah-train --config-name nas_ofa model="$model" experiment_id="$experiment_id"  nas="${CONFIG_2}" module.num_workers=8
}
function do_run {

    let model
    model=$1

    echo -e "$green"
    echo "Doing hannah config ${CONFIG_1} with grouping"
    echo -e "$yellow"
    cat "${PATH_TO_HANNAH}/conf/nas/${CONFIG_1}.yaml" |  sed 's/^/    /'
    let experiment_id=$2
    experiment_id=$2
    echo -e "$green"
    echo "using id '$experiment_id' with model '$model'"
    #hannah-train --config-name nas_ofa model="$model" experiment_id="$experiment_id"  nas="${CONFIG_1}" module.num_workers=8
    echo -e $reset
    echo "hannah-train --config-name nas_ofa model="$model" experiment_id="$experiment_id"  nas="${CONFIG_1}" module.num_workers=8"
    hannah-train --config-name nas_ofa model="$model" experiment_id="$experiment_id"  nas="${CONFIG_1}" module.num_workers=8

    # no grouping
    echo -e "$green"
    echo "Doing hannah config ${CONFIG_2} without grouping"
    echo -e "$yellow"
    cat "${PATH_TO_HANNAH}/conf/nas/${CONFIG_2}.yaml" |  sed 's/^/    /'
    echo -e "$green"
    experiment_id=$3
    echo "using id '$experiment_id' with model '$model'"
    echo -e $reset

    echo "hannah-train --config-name nas_ofa model="$model" experiment_id="$experiment_id"  nas="${CONFIG_2}" module.num_workers=8"
    hannah-train --config-name nas_ofa model="$model" experiment_id="$experiment_id"  nas="${CONFIG_2}" module.num_workers=8
}

#set -e
echo "Running now train batch:"
echo $(date)
do_run_normal
echo "1:Done"
echo "Running now quantazied version:"
do_run_quant
echo "2:Done"
echo "Train batch complete"
ls -la $PATH_TO_HANNAH/trained_models
# TODO some ana tools for the csv files
