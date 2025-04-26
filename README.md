# ** Börsenkurs Vorhersage **  
**Using ML and data science to analyze emission factors and pinpointing technical characteristics of cars which contribute to pollution.**

==============================

## **Table of Contents**  
1. [Overview](#overview)  
2. [Project Structure](#project-structure)  
3. [Dataset](#dataset)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Results](#results)  
7. [Contributing](#contributing)  
8. [License](#license)  

==============================

## **1. Overview**  


==============================


## **2. Project Structure**
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │    ├── external/  # Rohdaten von APIs (z.B. AlphaVantage, Yahoo Finance) [1][3]
    │    ├── raw/       # Unveränderte Originaldaten
    │    ├── processed/ # Bereinigte Daten für Modelleingaben
    │    └── interim/   # Zwischendatensätze während der Transformation
    
    api/
├── client.py         # API-Client für Datenabruf (Authentifizierung, Request-Handling)
├── data_schemas.py   # Datenvalidierungsschemata
└── endpoints/        # Konfiguration spezifischer Finanz-APIs [1][5]

src/
├── preprocessing/    # Feature-Engineering und Datenpipelining
├── models/
│   ├── train.py      # Modelltraining mit Hyperparameter-Tuning
│   └── predict.py    # Echtzeitvorhersagen [1][4]
├── evaluation/       # Backtesting und Performance-Metriken
└── utils/            # Hilfsfunktionen für Datentransformation

notebooks/            # Jupyter-Notebooks für EDA und Prototyping [3][4]
models/               # Serialisierte Modelle (Pickle/Joblib)
config/               # YAML/JSON-Konfigurationen für API-Keys und Modelle
requirements.txt      # Python-Abhängigkeiten

tests/
├── unit/       # Modul-Tests
└── integration/ # API- und Modellintegrationstests
reports/        # Performance-Metriken und Visualisierungen [4]


####### Original Data Struktur 
    
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports which summarize the project
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py

==============================

## **3. Dataset**
The project uses the EU (EEA) dataset "CO2 emissions from new passenger cars" from 2010-2023, covering 30 countries. Originally over 16 GB, it was reduced via data transformation. [Source](https://www.eea.europa.eu/data-and-maps/data/co2-emissions-from-new-passenger-cars-1)

==============================

## **4. Installation**
1. Clone the repository:
    ```bash
    git clone https://github.com/TilloStralka/CO2_Emission_Predictor.git
    cd CO2_Emission_Predictor
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

==============================

## **5. Usage**

#### **5.1. Data Preparation**
The dataset required for this project is not included in the repository because it is listed in the `.gitignore` file.  

##### **5.1.1. Download the Dataset**  
Please download the original dataset from the specified source and save it in the `data/raw/` folder. Ensure the file is named as required for seamless processing.

##### **5.1.2. Process the Data**  
Use the Jupyter Notebook located in the `notebooks/` folder to process the data:  
- Notebook: `DataPipeline.ipynb`  
- **Default Settings:** The callable functions in this notebook have default parameter values, but these can be adjusted as needed.  
- **Further Details:** For more in-depth information about the processing pipeline, refer to **Report #3** in the `reports/` folder.

##### **5.1.3. Finalized Dataset**  
After processing, the finalized dataset will have the following properties:
- **Name:** `minimal_withoutfc_dupesdropped_frequencies_area_removedskew_outliers3_0_NoW_tn20_mcp00.10.parquet` (cryptic title)  
- **Size:** 2,000,450 rows, 56 columns  
- **Memory Usage:** 282.4 MB  

#### **5.2. Modeling**
To perform the modeling, use the notebook:  
- Notebook: `ModelPipeline.ipynb`  
- This notebook contains all the necessary steps, fully commented for clarity.  

#### **5.3. Utilities**
The functions called in the notebooks are implemented in the `utils_co2.py` file. You can review this file for additional details about the underlying implementation.

#### **5.4. Trained Models**
After training, the trained models are saved in the `models/` folder for future use.  

---

#### **5.5. Additional Notes**
For any modifications to the processing or modeling pipeline, ensure that changes are consistent across all related files. For further assistance, consult the documentation and comments within the notebooks.

==============================

## **6. Results**

### Model Performance Metrics

The following metrics were evaluated across three models:

- **XG Boost** (Best Overall Performance):
    - Test MSE (scaled): 0.02
    - Test MSE: 25.1
    - Test RMSE: 5.0
    - Test MAE: 3.7
    - Test R²: 0.97
    - CV R²: 0.97
    - Training time: 1.41 mins

- **Dense Neural Network**:
    - Test MSE (scaled): 0.06
    - Test MSE: 64.0
    - Test RMSE: 8.0
    - Test MAE: 6.2
    - Test R²: 0.93
    - Training time: 0.26 mins

- **Linear Regression**:
    - Test MSE (scaled): 0.12
    - Test MSE: 120.5
    - Test RMSE: 11.0
    - Test MAE: 8.3
    - Test R²: 0.87
    - CV R²: 0.87
    - Training time (mins): 7.07

### Model Analysis

1. **XGBoost** emerged as the optimal choice, demonstrating:
    - Superior predictive accuracy with lowest error metrics
    - Excellent handling of non-linear relationships
    - Strong balance between performance and computational efficiency

2. **Dense Neural Network** showed promise with:
    - Fast training speed
    - Strong R² score of 0.93
    - Potential for improvement through overfitting prevention

3. **Linear Regression**, while most interpretable:
    - Provided baseline performance
    - Limited by linear assumptions
    - Required longest training time

### Model Performance

- **XGBoost**:
    - Achieved the best predictive accuracy.
    - Lowest Mean Squared Error (MSE) and highest R-squared.
    - Effectively captures non-linear relationships.
    - Provides an optimal balance of performance and efficiency.

- **Dense Neural Network (DNN)**:
    - A strong contender with fast training and high R-squared.
    - Requires further fine-tuning to prevent overfitting.

- **Linear Regression**:
    - Offers simplicity and interpretability.
    - Underperforms relative to XGBoost and DNN.

**Recommendation**: XGBoost 

### Outlook and Improvements

- **Ensemble Learning**: Combining Linear Regression, XGBoost, and Dense Neural Networks for improved predictions.
- **Hyperparameter Tuning**: Leveraging techniques like Bayesian Optimization for XGBoost and DNN.
- **Advanced Architectures**: Experimenting with CNNs or Residual Connections to boost model performance.
- **Data Augmentation**: Using methods like SMOTE to address class imbalances.


==============================

## **7. Contributing**

This project was developed as part of the Data Scientist certification program at DataScientest.com, certified by Université Paris Panthéon-Sorbonne. The following team members contributed equally to this project:

- [Operator-12](https://github.com/Operator-12)
- [Alexander Peca](https://github.com/Alexander-Peca)
- [Shanthi Dev](https://github.com/ShanthiDev)

==============================

## **8. License**

This project is licensed under the MIT License.