#!/bin/bash


SEED=1

######BACKBONE######

python3 ./cf.py --seed=$SEED --dataset=micro_video --model-name=mf --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001
python3 ./seq_rec.py --seed=$SEED --dataset=micro_video --model-name=grurec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2
python3 ./seq_rec.py --seed=$SEED --dataset=micro_video --model-name=sasrec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --max-seq-len=50 --dropout=0.2 --n-heads=1
python3 ./seq_rec_tisasrec.py --seed=$SEED --dataset=micro_video --model-name=tisasrec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --max-seq-len=50 --dropout=0.2 --n-heads=1 --time-span=512
python3 ./seq_rec.py --seed=$SEED --dataset=micro_video --model-name=fearec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --max-seq-len=50 --dropout=0.2 --n-heads=1
python3 ./seq_rec.py --seed=$SEED --dataset=micro_video --model-name=bsarec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --max-seq-len=50 --dropout=0.2 --n-heads=1 --alpha=0.7 --c=1 

python3 ./cf.py --seed=$SEED --dataset=ml-1m --model-name=mf --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001
python3 ./seq_rec.py --seed=$SEED --dataset=ml-1m --model-name=grurec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2
python3 ./seq_rec.py --seed=$SEED --dataset=ml-1m --model-name=sasrec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --max-seq-len=50 --dropout=0.2 --n-heads=1
python3 ./seq_rec_tisasrec.py --seed=$SEED --dataset=ml-1m --model-name=tisasrec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --max-seq-len=50 --dropout=0.2 --n-heads=1 --time-span=2048
python3 ./seq_rec.py --seed=$SEED --dataset=ml-1m --model-name=fearec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --max-seq-len=50 --dropout=0.2 --n-heads=1
python3 ./seq_rec.py --seed=$SEED --dataset=ml-1m --model-name=bsarec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --max-seq-len=50 --dropout=0.2 --n-heads=1 --alpha=0.7 --c=1 

python3 ./cf.py --seed=$SEED --dataset=kuairand --model-name=mf --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001
python3 ./seq_rec.py --seed=$SEED --dataset=kuairand --model-name=grurec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2
python3 ./seq_rec.py --seed=$SEED --dataset=kuairand --model-name=sasrec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --max-seq-len=50 --dropout=0.2 --n-heads=1
python3 ./seq_rec_tisasrec.py --seed=$SEED --dataset=kuairand --model-name=tisasrec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --max-seq-len=50 --dropout=0.2 --n-heads=1 --time-span=512
python3 ./seq_rec.py --seed=$SEED --dataset=kuairand --model-name=fearec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --max-seq-len=50 --dropout=0.2 --n-heads=1
python3 ./seq_rec.py --seed=$SEED --dataset=kuairand --model-name=bsarec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --max-seq-len=50 --dropout=0.2 --n-heads=1 --alpha=0.9 --c=1


######PROPOSED######

python3 ./debiased_cf.py --seed=$SEED --dataset=micro_video --model-name=mf --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --tau=0.1 --gamma=0.9 --eta=0.3
python3 ./debiased_seq_rec.py --seed=$SEED --dataset=micro_video --model-name=grurec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --tau=0.1 --gamma=0.5 --eta=0.3
python3 ./debiased_seq_rec.py --seed=$SEED --dataset=micro_video --model-name=sasrec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --max-seq-len=50 --dropout=0.2 --n-heads=1 -tau=0.1 --gamma=0.7 --eta=0.3
python3 ./debiased_seq_rec_tisasrec.py --seed=$SEED --dataset=micro_video --model-name=tisasrec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --max-seq-len=50 --dropout=0.2 --n-heads=1 --time-span=512 --tau=0.1 --gamma=0.3 --eta=0.3
python3 ./debiased_seq_rec.py --seed=$SEED --dataset=micro_video --model-name=fearec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --max-seq-len=50 --dropout=0.2 --n-heads=1 --tau=0.1 --gamma=0.1 --eta=0.3
python3 ./debiased_seq_rec.py --seed=$SEED --dataset=micro_video --model-name=bsarec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --max-seq-len=50 --dropout=0.2 --n-heads=1 --alpha=0.7 --c=1 --tau=0.1 --gamma=0.5 --eta=0.3

python3 ./debiased_cf.py --seed=$SEED --dataset=ml-1m --model-name=mf --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --tau=0.1 --gamma=0.9 --eta=0.7
python3 ./debiased_seq_rec.py --seed=$SEED --dataset=ml-1m --model-name=grurec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --tau=0.1 --gamma=0.9 --eta=0.5
python3 ./debiased_seq_rec.py --seed=$SEED --dataset=ml-1m --model-name=sasrec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --max-seq-len=50 --dropout=0.2 --n-heads=1 --tau=0.1 --gamma=0.5 --eta=0.5
python3 ./debiased_seq_rec_tisasrec.py --seed=$SEED --dataset=ml-1m --model-name=tisasrec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --max-seq-len=50 --dropout=0.2 --n-heads=1 --time-span=2048 --tau=0.1 --gamma=0.9 --eta=0.5
python3 ./debiased_seq_rec.py --seed=$SEED --dataset=ml-1m --model-name=fearec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --max-seq-len=50 --dropout=0.2 --n-heads=1 --tau=0.1 --gamma=0.1 --eta=0.5
python3 ./debiased_seq_rec.py --seed=$SEED --dataset=ml-1m --model-name=bsarec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --max-seq-len=50 --dropout=0.2 --n-heads=1 --alpha=0.7 --c=1 --tau=0.1 --gamma=0.7 --eta=0.7

python3 ./debiased_cf.py --seed=$SEED --dataset=kuairand --model-name=mf --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --tau=0.1 --gamma=0.1 --eta=0.5
python3 ./debiased_seq_rec.py --seed=$SEED --dataset=kuairand --model-name=grurec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --tau=0.1 --gamma=0.3 --eta=0.7
python3 ./debiased_seq_rec.py --seed=$SEED --dataset=kuairand --model-name=sasrec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --max-seq-len=50 --dropout=0.2 --n-heads=1 --tau=0.1 --gamma=0.9 --eta=0.5
python3 ./debiased_seq_rec_tisasrec.py --seed=$SEED --dataset=kuairand --model-name=tisasrec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --max-seq-len=50 --dropout=0.2 --n-heads=1 --time-span=512 --tau=0.1 --gamma=0.7 --eta=0.5
python3 ./debiased_seq_rec.py --seed=$SEED --dataset=kuairand --model-name=fearec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --max-seq-len=50 --dropout=0.2 --n-heads=1 --tau=0.1 --gamma=0.3 --eta=0.5
python3 ./debiased_seq_rec.py --seed=$SEED --dataset=kuairand --model-name=bsarec --epochs=500 --recdim=128 --batch-size=32384 --contrast-size=16 --lr=0.001 --depth=2 --max-seq-len=50 --dropout=0.2 --n-heads=1 --alpha=0.9 --c=1 --tau=0.1 --gamma=0.9 --eta=0.5
