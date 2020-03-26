#!/bin/bash
# Copyright 2020 Beijing Xiaomi Intelligent Technology Co.,Ltd  Weiji Zhuang
echo "$0 $@ : PID:$$ begining at `hostname -f` when `date +%Y-%m-%d-%T`"

nj=200
stage=1

[ -f ./cmd.sh ] && . ./cmd.sh || { echo "Error: No cmd.sh"; exit 1; }
[ -f ./path.sh ] && . ./path.sh || { echo "Error: No path.sh"; exit 1; }
[ -f ./utils/parse_options.sh ] && . ./utils/parse_options.sh || { echo "Error:No uparse_options.sh"; exit 1; }

if [ $stage -le 0 ]; then
utils/copy_data_dir.sh --utt-suffix "-Android" data/orig/Android data/train_Android
utils/copy_data_dir.sh --utt-suffix "-IOS" data/orig/IOS data/train_IOS
utils/copy_data_dir.sh --utt-suffix "-Recorder" data/orig/Recorder data/train_Recorder
utils/combine_data.sh data/train_all data/train_Android data/train_IOS data/train_Recorder
utils/copy_data_dir.sh data/train_all data/train
utils/copy_data_dir.sh data/train_all data/test
# total : 18255
head -n 17000 data/train_all/utt2spk >data/train/utt2spk
tail -n 1255 data/train_all/utt2spk >data/test/utt2spk
utils/fix_data_dir.sh data/train
utils/fix_data_dir.sh data/test
local/make_spectrogram.sh --nj $nj --cmd "$train_cmd" --spect-config conf/spect.conf --compress false data/train
steps/compute_cmvn_stats.sh --fake  data/train
local/make_spectrogram.sh --nj $nj --cmd "$train_cmd" --spect-config conf/spect.conf --compress false data/test
steps/compute_cmvn_stats.sh --fake  data/test

utils/copy_data_dir.sh --utt-suffix "-Android" data/orig/spk data/train_Android_target
utils/copy_data_dir.sh --utt-suffix "-IOS" data/orig/spk data/train_IOS_target
utils/copy_data_dir.sh --utt-suffix "-Recorder" data/orig/spk data/train_Recorder_target
utils/combine_data.sh data/train_all_target data/train_Android_target data/train_IOS_target data/train_Recorder_target
utils/copy_data_dir.sh data/train_all_target data/train_target
utils/copy_data_dir.sh data/train_all_target data/test_target
# total : 18255
head -n 17000 data/train_all_target/utt2spk >data/train_target/utt2spk
tail -n 1255 data/train_all_target/utt2spk >data/test_target/utt2spk
utils/fix_data_dir.sh data/train_target
utils/fix_data_dir.sh data/test_target
local/make_spectrogram.sh --nj $nj --cmd "$train_cmd" --spect-config conf/spect.conf --compress false data/train_target
steps/compute_cmvn_stats.sh --fake  data/train_target
local/make_spectrogram.sh --nj $nj --cmd "$train_cmd" --spect-config conf/spect.conf --compress false data/test_target
steps/compute_cmvn_stats.sh --fake  data/test_target
fi

nnet3_affix=_dae
dnn_affix=_rn
train_stage=-10
remove_egs=false
srand=0
cmvn_options="--norm-means=false --norm-vars=false"
common_egs_dir=exp/nnet3_dae/tdnn_bn/egs
dir=exp/nnet3${nnet3_affix}/tdnn${dnn_affix}
data_dir=data/train
target_dir=data/train_target

# prepare neural network
if [ $stage -le 1 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";
  num_inputs=$(feat-to-dim scp:${data_dir}/feats.scp -)
  num_targets=$(feat-to-dim scp:${target_dir}/feats.scp -)

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=$num_inputs name=input
  relu-renorm-layer name=dnn1 dim=1024 input=Append(-5,-4,-3,-2,-1,0,1,2,3,4,5)
  relu-renorm-layer name=dnn2 dim=1024
  relu-renorm-layer name=dnn3 dim=1024
  relu-renorm-layer name=dnn4 dim=1024
  output-layer name=output dim=$num_targets include-log-softmax=False objective-type=quadratic
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

# training
if [ $stage -le 2 ]; then
  steps/nnet3/train_raw_dnn.py \
    --stage=$train_stage \
    --cmd="$cuda_cmd" \
    --feat.cmvn-opts="$cmvn_options" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=40 \
    --trainer.samples-per-iter=400000 \
    --trainer.optimization.num-jobs-initial=1 \
    --trainer.optimization.num-jobs-final=1 \
    --trainer.optimization.initial-effective-lrate=0.0001 \
    --trainer.optimization.final-effective-lrate=0.00001 \
    --trainer.optimization.minibatch-size=128,64 \
    --egs.dir="$common_egs_dir" \
    --egs.cmd="$egs_cmd" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=true \
    --feat-dir=$data_dir \
    --targets-scp=$target_dir/feats.scp \
    --dir=$dir  || exit 1;
fi

# test
if [ $stage -le 3 ]; then
  echo "start denoising ..."
  nnet3-compute --verbose=2 --use-gpu=no \
    $dir/final.raw scp:data/test/feats.scp \
    scp,ark:data/test_target/target_feats${dnn_affix}.scp,data/test_target/target_feats${dnn_affix}.ark
fi

# plot
if [ $stage -le 4 ]; then
  python local/plot_multi_feats.py 3 \
    data/test/feats.scp data/test_target/feats.scp \
    data/test_target/target_feats${dnn_affix}.scp figure${dnn_affix}
  python steps/nnet3/report/generate_plots.py $dir $dir/plot
fi
