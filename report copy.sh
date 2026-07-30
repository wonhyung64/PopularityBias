#!/bin/bash


ENV=/home1/wonhyung64/anaconda3/envs/openmmlab/bin/python3
DATADIR=/home1/wonhyung64/Github/ldr_rec/data


python ./debiased_cf.py --model-name=mf --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.9 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --eta=0.3
python ./debiased_seq_rec.py --model-name=grurec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.5 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --eta=0.3
python ./debiased_seq_rec.py --model-name=sasrec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.7 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --eta=0.3
python ./debiased_seq_rec_tisasrec.py --model-name=tisasrec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.3 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --eta=0.3
python ./debiased_seq_rec.py --model-name=fearec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.1 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --eta=0.3
python ./debiased_seq_rec.py --model-name=bsarec --dataset=micro_video --seed=1 --tau=0.1 --lambda1=0.5 --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --eta=0.3

python ./debiased_cf.py --model-name=mf --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.9 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --eta=0.7
python ./debiased_seq_rec.py --model-name=grurec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.9 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --eta=0.5
python ./debiased_seq_rec.py --model-name=sasrec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.5 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --eta=0.5
python ./debiased_seq_rec_tisasrec.py --model-name=tisasrec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.9 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --eta=0.5
python ./debiased_seq_rec.py --model-name=fearec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.1 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --eta=0.5
python ./debiased_seq_rec.py --model-name=bsarec --dataset=ml-1m --seed=1 --tau=0.1 --lambda1=0.7 --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.7 --c=1 --epochs=500 --eta=0.7

python ./debiased_cf.py --model-name=mf --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.1 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --eta=0.5
python ./debiased_seq_rec.py --model-name=grurec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.3 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --eta=0.7
python ./debiased_seq_rec.py --model-name=sasrec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.9 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --eta=0.5
python ./debiased_seq_rec_tisasrec.py --model-name=tisasrec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.7 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --eta=0.5
python ./debiased_seq_rec.py --model-name=fearec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.3 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500 --eta=0.5
python ./debiased_seq_rec.py --model-name=bsarec --dataset=kuairand --seed=1 --tau=0.1 --lambda1=0.9 --pair-reset-interval=5 --evaluate-interval=500 --alpha=0.9 --c=1 --epochs=500 --eta=0.5
