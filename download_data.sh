mkdir -p data && cd data
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -sL -o $f.gz https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/$f.gz
done