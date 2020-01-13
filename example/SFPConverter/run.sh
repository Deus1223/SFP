# -p fraction bit length
# -m path to trained model
# -td path to training data
# -vd path to test data

make clean

# make test
# ./test -m "./hdf5_convert/model.txt" -td "../data/train_data.txt" -vd "../data/test_data.txt"

make
./SFP -p 15 -m "../FormatConverter/model_hdf5.txt" -td "../Data/dnnin/train_data.txt" -vd "../Data/dnnin/test_data.txt"