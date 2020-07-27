# Swiss Life Data Science Technical interview 

## Description
Machine learning model for predict apply acceptance for an open job position.

All the project has been developed with **pipenv virtual environment** on **visual studio code** text editor (on **MAC OS X**). Notebooks will use the current activated environment as kernel. If environment fail to start because of any issues provided by unmatched dependencies, feel free to go to this **github repo** where codes and results are currently deposited on.
```
https://github.com/hadjigagny93/Swiss-Life-Data-Science-Test
```

### 1. Setup virtual environment 
Download project folder and go to root directory
Make sure you have conda env with **python 3.8.4** and pipenv installed on it, you can check it with. 
```
conda activate 
python3 --version
pip list 
```
If so, run the following commands 
- create virtual environment and activate it
- deactivate conda env``
- install pipenv 
```

pipenv shell 
conda deactivate 
pip install pipenv

```
After that, run for installing dependencies

```
pipenv install
```

### 2. Data
in data folder, original data are splited into train and test data with this command, you do not need to do it if train and test csv files are already in the folder (feel free to test anyway). 
```
python src/infrastructure/scripts.py
```


### 3. Test code 
Go to notebook folder and run the cells!

### 4. Results 

You can find summary.pdf file in doc folder 