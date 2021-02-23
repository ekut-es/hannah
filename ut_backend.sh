TEDA=/local/palomero/projects/teda
LOAD=false

PERIOD=1000
MACRO_TYPE=RTL
CHANNELS=16

for DIM in 8
do
  for BIT in 8
  do
    if [ "$LOAD" = true ]; then
      rm -rf ${TEDA}/output
      #cp -r ${TEDA}/rtl/tc-resnet8-accelerator/results/${DIM}x${DIM}_W${BIT}B${BIT}F${BIT} ${TEDA}/output
    fi

    .venv/bin/python3 -m speech_recognition.train \
      trainer.gpus=Null \
      trainer.max_epochs=1 \
      compress.fold_bn=0.05 \
      trainer.limit_train_batches=0.1 \
      trainer.limit_val_batches=0.1 \
      trainer.limit_test_batches=0.01 \
      model=tc-res8 \
      model.block1_output_channels=$CHANNELS \
      model.block2_output_channels=$CHANNELS \
      model.block3_output_channels=$CHANNELS \
      features.n_mfcc=$CHANNELS \
      compress=fp_${BIT}_${BIT}_${BIT} \
      normalizer=fixedpoint \
      normalizer.normalize_bits=$BIT \
      normalizer.normalize_max=$((2 ** $BIT)) \
      backend=trax_ut \
      backend.standalone=True \
      backend.rtl_simulation=True \
      backend.synthesis=False \
      backend.postsyn_simulation=False \
      backend.power_estimation=False \
      backend.num_inferences=1 \
      backend.cols=$DIM \
      backend.rows=$DIM \
      backend.bw_f=$BIT \
      backend.bw_w=$BIT \
      backend.bw_b=$BIT \
      backend.period=$PERIOD \
      backend.macro_type=$MACRO_TYPE

    if [ "$LOAD" = true ]; then
      find ${TEDA}/output -name "*.vcd.gz" -type f -delete
      rm -rf ${TEDA}/rtl/tc-resnet8-accelerator/results/${DIM}x${DIM}_W${BIT}B${BIT}F${BIT}
      mv ${TEDA}/output ${TEDA}/rtl/tc-resnet8-accelerator/results/${DIM}x${DIM}_W${BIT}B${BIT}F${BIT}
    fi
  done
done
