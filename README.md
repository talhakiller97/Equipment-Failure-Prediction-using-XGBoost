

# 🚧 Equipment Failure Prediction using XGBoost

This project implements a predictive maintenance system for industrial pumps using machine learning. By analyzing time-series sensor data, it forecasts whether equipment failure is likely to occur within the next 24 hours and generates a CSV-based maintenance schedule.

---

## 🔍 Problem Statement

Unplanned equipment failures in manufacturing can be extremely costly. This system predicts failures in advance by learning from historical sensor readings, enabling timely maintenance and minimizing downtime.

---

## 📦 Features

- Predicts **next 24-hour failure** risk  
- Uses **XGBoost Classifier** for high accuracy  
- Outputs a **maintenance schedule CSV**  
- Handles **time-series preprocessing** and class imbalance  
- Visualizes **ROC curve**, **confusion matrix**, and metrics  
- Built for future integration into **real-time monitoring systems**

---

## 🧠 Machine Learning Pipeline

- **Model**: XGBoost (with early stopping and class weighting)  
- **Target**: Binary classification (Failure in next 24 hours: 1 or 0)  
- **Preprocessing**:
  - Timestamp parsing
  - Rolling averages
  - Feature scaling
  - One-hot encoding (for categorical variables)
- **Evaluation Metrics**:
  - Accuracy
  - Precision, Recall, F1-score
  - ROC-AUC score

---

## 📁 Project Structure

```

Equipment-Failure-Prediction-using-XGBoost/
├── data/
│   └── predictive\_maintenance.csv
├── notebooks/
│   └── EDA\_and\_Modeling.ipynb
├── outputs/
│   ├── raw\_data\_with\_predictions.csv
│   └── failures\_next\_24\_hours.csv
├── src/
│   ├── preprocess.py
│   ├── model.py
│   └── schedule\_generator.py
├── requirements.txt
├── README.md
└── LICENSE

````

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/talhakiller97/Equipment-Failure-Prediction-using-XGBoost.git
cd Equipment-Failure-Prediction-using-XGBoost
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline

```bash
python src/preprocess.py
python src/model.py
python src/schedule_generator.py
```

### 4. View outputs

* `outputs/raw_data_with_predictions.csv`: full dataset with predicted failure labels
* `outputs/failures_next_24_hours.csv`: generated maintenance plan

---

## 💡 Use Cases

* Smart Manufacturing
* Industry 4.0 / IIoT
* Pump Health Monitoring
* Condition-Based Maintenance Systems

---

## 🔮 Future Improvements

* Live sensor feed simulation
* Real-time dashboard using Streamlit or Dash
* SHAP/LIME model interpretability
* Multi-class fault type prediction

---

## 👤 Author

**Talha Saeed**
Data Scientist & Machine Learning Engineer
[GitHub](https://github.com/talhakiller97)
---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

