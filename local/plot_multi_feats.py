#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2020 Xiaomi Corporation (author: Weiji Zhuang)
# 2020/3/25

# Apache 2.0

import csv
import os
import sys
import kaldi_io
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')


def read_feats_ark(feat_ark):
    mat = kaldi_io.read_mat(feat_ark)
    return mat


def scp2dict(scp):
    '''Convert a list of tuples read from the wavscp file to a dictionary.'''
    out = {}
    for row in scp:
        out[row[0]] = row[1]
    return out


def read_csv(filename):
    '''Read a CSV (CTM) file.'''
    with open(filename, 'r') as fileobj:
        out = list(csv.reader(fileobj, delimiter=' ', skipinitialspace=True))
    return out


def main():
    if len(sys.argv) < 3:
        print("Usage: %s <figure-n> <scp-1> ... <scp-n> <out-dir>\n" % (
            sys.argv[0]))
        exit(1)
    print("Ploting ...\n")
    tot_num_figure = int(sys.argv[1])
    feat_scp = sys.argv[2]
    out_dir = sys.argv[len(sys.argv) - 1]
    if not os.path.exists(os.path.join(out_dir)):
        os.makedirs(os.path.join(out_dir))
    feat_dict = [scp2dict(read_csv(feat_scp))]

    for num in range(3, len(sys.argv) - 1):
        feat_dict.append(scp2dict(read_csv(sys.argv[num])))

    num_figure = 1
    for utt in feat_dict[0].keys():
        fig = plt.figure(figsize=(20, 12))
        for num in range(0, len(feat_dict)):
            feats = read_feats_ark(feat_dict[num][utt])
            plt.subplot(len(feat_dict), 1, num + 1)
            heatmap = plt.pcolor(feats.T)
            fig.colorbar(mappable=heatmap)
            plt.xlabel('Frame')
            plt.ylabel('Feat')
            # plt.title(title)
            # plt.tight_layout()
        plt.savefig(out_dir + '/' + utt + '.png')
        num_figure += 1
        if num_figure > tot_num_figure:
            break
    print("Stored in %s" % (out_dir))


if __name__ == '__main__':
    main()
