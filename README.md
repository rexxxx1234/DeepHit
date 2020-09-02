# DeepHit


## How to run   
First, install dependencies   
```bash
# clone project   
git clone git@github.com:zydou/DeepHit.git   

# install project   
cd DeepHit
conda env create -f environment.yml
 ```   
 
 Next, activate the environment and run it.   
 ```bash
conda activate deephit

# run module: 
python main.py --data_name SYNTHETIC --gpus 1 --precision 16 --batch_size 256 --max_epochs 100

# check `models/deephit.py` for more args
```
