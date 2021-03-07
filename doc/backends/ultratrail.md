# Ultratrail backend

To use the ultratrail backend use

1. Install teda: https://atreus.informatik.uni-tuebingen.de/ties/t-rax/teda
2. Set environment variable `TEDA_HOME` to path of teda checkout

To run it use:

     python -m speech_recognition.train compress=fp_8_8_8 backend=trax_ut

## Configuration

teda_dir
: ${env:TEDA_HOME,/local/frischkn/neural_networks/teda}

backend_dir
: "${backend.teda_dir}/rtl/tc-resnet8-accelerator"

standalone
: False

rtl_simulation
: False

synthesis
: False

postsyn_simulation
: False

power_estimation
: False

num_inferences
: 1

bw_w
: 8

bw_b
: 8

bw_f
: 8

cols
: 8

rows
: 8

period
: 1000

macro_type
: "RTL"

use_acc_statistic_model
: True

use_acc_analytical_model
: False
