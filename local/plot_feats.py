#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2020 Xiaomi Corporation (author: Weiji Zhuang)
# 2020/3/24

# Apache 2.0


import kaldi_io
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')


def plot_feats(feats, feat_type, title, file_name):
    """Draw the spectrogram picture
        :param spec: a feature_dim by num_frames array(real)
        :param note: title of the picture
        :param file_name: name of the file
    """
    fig = plt.figure(figsize=(20, 6))
    heatmap = plt.pcolor(feats)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(ms)')
    plt.ylabel(feat_type)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(file_name)


def read_all_data(feat_scp):
    feat_fid = open(feat_scp, 'r')
    feat = feat_fid.readlines()
    feat_fid.close()
    id_list = []
    mat_list = []

    for i in range(len(feat)):
        id, ark = feat[i].split()
        mat = kaldi_io.read_mat(ark)
        id_list.append(id)
        mat_list.append(mat)
    return id_list, mat_list


def main():
    ids, feats = read_all_data('/home/storage04/zhuangweiji/workspace/x260remotespace/DAE/feats.scp')
    for i in range(len(ids)):
        plot_feats(feats[i].T, 'spectrogram', ids[i],
                   '/home/storage04/zhuangweiji/workspace/x260remotespace/DAE/tmp/'+ ids[i] +'.png')

if __name__ == '__main__':
    main()
