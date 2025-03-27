# Handwritten Text Recognition with PyTorch multiple GPU's

This model for *handwritten text recognition* implements CNN+LSTM with CTC loss and beam search CTC decoding. It is required to install [ctcdecode library](https://github.com/parlance/ctcdecode) along with other requirements. 

# Install the required package "ctcdecode"
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode

# Modify the setup.py to allow ngram order up to 10
sed -r "/-DKENLM_MAX_ORDER=/\
s/-DKENLM_MAX_ORDER=[0-9]+/-DKENLM_MAX_ORDER=10/" \
setup.py > aux
mv aux setup.py
pip install . #python setup.py install

cd ../..
