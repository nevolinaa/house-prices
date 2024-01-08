# house-prices
Predicting the price of a house based on its characteristics

## Initial configuration
**For usage:**  
```
pip install miniconda
conda create -n house_prices python=3.11
conda activate house_prices
```
- clone the repository
```
cd house-prices
poetry install
pre-commit install
pre-commit run -a
```

## Train model
```
python train.py
```

## Predict on test
```
python infer.py
```
