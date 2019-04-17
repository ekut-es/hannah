#!/bin/bash

source .tcml_config

time=""
partition=""
name=""
command=""

while [[ $# -gt 0 ]]
do
    key=$1
    
    case $key in
	--name) 
	    shift
	    name=$1
	    shift
	    ;;
	--time) 
	    shift
	    time=$1
	    shift
	    ;;
	--partition)
	    shift
	    partition=$1
	    shift
	    ;;
	*)    # unknown option
	    echo "Found unknown option: $key"
	    exit 1
	    ;;
    esac
done

command=$*

echo "Running on tcml-cluster"
echo "Config:"
echo "  Job Name:          $name"
echo "  Username:          $USER"
echo "  User Mail:         $MAIL"
echo "  Working directory: $WD"
echo "  Expected Time:     $time"
echo "  Cluster Partition: $partition"

