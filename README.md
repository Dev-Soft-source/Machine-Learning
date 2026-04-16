# Housing Price Prediction

This project predicts California housing prices using machine learning with Python and Scikit-learn.

## Project Structure

```text
housing-price-prediction/
│
├── data/
│   ├── raw/
│   │   └── housing.csv
│   └── processed/
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── model.pkl
│
├── outputs/
│   ├── figures/
│   └── reports/
│
├── app/
│   └── simple_ui.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── main.py
```

## Usage

Install dependencies and run the Streamlit app with:

```bash
pip install -r requirements.txt
streamlit run app/simple_ui.py
```
