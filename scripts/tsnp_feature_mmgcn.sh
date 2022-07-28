for seed in {1..3}; do
  for m in 'atv' 'av' 'at' 'tv'; do
    python3 train_mm.py --module=mmgcn --dataset=iemocap-cogmen-tsn-v+-4 --modality=$m --reimplement --seed=$seed
    python3 train_mm.py --module=mmgcn --dataset=iemocap-cogmen-robert-tsn-v+-4 --modality=$m --reimplement --seed=$seed
    python3 train_mm.py --module=mmgcn --dataset=iemocap-cogmen-sbert-tsn-v+-4 --modality=$m --reimplement --seed=$seed
    python3 train_mm.py --module=mmgcn --dataset=iemocap-cogmen-tsn-v+-6 --modality=$m --reimplement --seed=$seed
    python3 train_mm.py --module=mmgcn --dataset=iemocap-cogmen-robert-tsn-v+-6 --modality=$m --reimplement --seed=$seed
    python3 train_mm.py --module=mmgcn --dataset=iemocap-cogmen-sbert-tsn-v+-6 --modality=$m --reimplement --seed=$seed
  done
done
