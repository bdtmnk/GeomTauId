# GeomTauId
Geometric Learning for tau identification

# Maxwell instalation:

./source create -n py35_tau_id

source activate  py35_tau_id

conda install python=3.5\n

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

conda install cython

pip3 install --verbose --no-cache-dir torch-scatter --user

pip3 install --verbose --no-cache-dir torch-sparse --user

pip3 install --verbose --no-cache-dir torch-cluster --user

pip3 install torch-geometric --user

pip3 install uproot --user

pip3 install -U skorch --user

pip3 install torch torchvision --user
