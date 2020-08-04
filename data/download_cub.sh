#!/usr/bin/env bash
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar -zxvf CUB_200_2011.tgz
rm CUB_200_2011.tgz && rm attributes.txt
python split_CUB.py
rm -r CUB_200_2011