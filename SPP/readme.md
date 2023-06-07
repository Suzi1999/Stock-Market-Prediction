# Stock Price Prediction using LSTM

## Introduction

This project is about predicting the stock price of any NSE(National Stock Exchange) listed company using LSTM(Long Short Term Memory) model. The model is trained on the historical data of the company and then used to predict the stock price for the next 100 days.

## Requirements

- Python 3.6
- Keras
- Tensorflow
- Numpy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- Streamlit
- Yahoo Finance

## Installation

### Step 1: Clone the repository

```bash
git clone
```

### Step 2: Create a virtual environment

```bash
python3 -m venv spp-env
```

### Step 3: Activate the virtual environment

- For Windows :

```bash
spp-env\Scripts\activate
```

- For Linux :

```bash
source spp-env/bin/activate
```

### Step 4: Install the dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Run the Jupyter Notebook

```bash
pip install jupyter nbconvert

jupyter nbconvert --to script spp-ml.ipynb

python spp-ml.py
```

### Step 2: Run the Streamlit app

```bash
streamlit run app.py
```

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Author

- [Bandana Bharti](https://github.com/Bandana320)
