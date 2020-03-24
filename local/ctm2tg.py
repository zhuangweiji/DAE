#!/usr/bin/env python
# Copyright 2019 Beijing Xiaomi Intelligent Technology Co.,Ltd  Weiji Zhuang
'''
Convert Kaldi's CTM alignment output to Praat's TextGrid format.
'''

import csv
import sys
import os
import shutil
import re
from praatio import tgio


def readCSV(filename):
    '''Read a CSV (CTM) file.'''
    with open(filename, 'rb') as fileobj:
        out = list(csv.reader(fileobj, delimiter=' ', skipinitialspace=True))
    return out


def csv2tgdict(ctmlist):
    '''Convert a list of tuples read from the CTM file to a TextGrid dictionary.'''
    out = {}
    for row in ctmlist:
        if row[0] not in out:
            out[row[0]] = []
        segment = (row[2],
                   str(float(row[2]) + float(row[3])),
                   row[4].split('_')[0])
        out[row[0]].append(segment)
    return out


def wavscp2dict(wavscp):
    '''Convert a list of tuples read from the wavscp file to a dictionary.'''
    out = {}
    for row in wavscp:
        out[row[0]] = row[1]
    return out


def main():
    '''Convert CTM alignment files to Praat's TextGrid format.
    Args:
    wavscp -- path to the directory containing speech wav files
    outdir -- path to output the textgrid files in
    '''
    if (len(sys.argv) < 3):
        print "Usage: %s <wav.scp> <outdir> <cmt1> <cmt2> ... <cmtn>" % (sys.argv[0])
        exit(1)
    print "Converting ctm files to Praat Textgrids...\n",
    wavscp = sys.argv[1]
    outdir = sys.argv[2]
    #absOutDir = os.path.abspath(outdir)
    wavdict = wavscp2dict(readCSV(wavscp))
    if not os.path.exists(os.path.join(outdir)):
        os.makedirs(os.path.join(outdir))

    for utt in wavdict.keys():
        tg = tgio.Textgrid()
        for num in range(3, len(sys.argv)):
            ctmcsv = readCSV(sys.argv[num])
            tgdict = csv2tgdict(ctmcsv)
            if not os.path.isfile(wavdict[utt]):
                print "%s not exist!" % (wavdict[utt])
                break
            else:
                fpath, fname = os.path.split(wavdict[utt])
                shutil.copyfile(wavdict[utt], os.path.join(outdir, fname))
                intervalTier = tgio.IntervalTier(
                    sys.argv[num], tgdict[utt], 0, pairedWav=wavdict[utt])
                tg.addTier(intervalTier)
        tg.save(os.path.join(outdir, utt + '.TextGrid'))

    print "stored in " + outdir


if __name__ == '__main__':
    main()
