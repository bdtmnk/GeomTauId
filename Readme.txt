In order to Train DPF CNN:
python Trainer.py --config=config_maxwell.ini
DeepSets:
python Trainer.py --config=config_deepsets.ini

#Evaluate:
python Evaluate.py --config=config.ini --file=DYInc_2016_25_tree.root
#For the BinaryClassification Evaluation

python EvaluateMCC.py --config=config.ini --file=DYInc_2016_25_tree.root
#For the MultyClassification Evaluation

config any config with the same structure as config_maxwell.ini
file root file from validation directory
Its is possible to run it using batch system
mkdir jobs
cp HTS_submit.sh bss jobs
./submit_eval_CNN.sh validation.txt config_maxwell.ini


