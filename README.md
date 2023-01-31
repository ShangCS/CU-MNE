Requirements
- Python 3.6
- numpy>=1.19.5
- scipy>=1.5.4
- scikit-learn>=0.24.1
- tqdm>=4.59.0
- torch>=1.6.0 
- torchvision>=0.7.0

For PyTorch, please install the version compatible with your machine.

Data
The data can be downloaded from (https://www.dropbox.com/s/48oe7shjq0ih151/data.tar.gz?dl=0).
Each dataset is a dictionary containing the following keys:
- `train_idx`, `val_idx` and `test_idx` are indices for training, validation and testing; 
`label` corresponds to the labels of the nodes;
- the layer names of the dataset: e.g., `MAM` and `MDM` for the `imdb` dataset.

Run
1. Download the ACM, IMDB, and Amazon data from (https://www.dropbox.com/s/48oe7shjq0ih151/data.tar.gz?dl=0) and put it to the folder data.
2. Specify the arguments in the main.py.
3. Run the code by 'python main.py'.

Results
./Results/best_model/ stores the best result for each dataset.
