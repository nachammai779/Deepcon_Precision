DEEPCON Using PRECISION Features as Input
Trained and Validated using the 3456 proteins in the DeepCov dataset with the precision features (441 channels) as input
Predict
python ../deepcon-precision.py --aln ./16pkA0.aln --rr ./16pkA0.rr
Evaluate
./coneva-lite.pl -pdb ./16pkA.pdb -rr ./16pkA0.rr
