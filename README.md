# ** Börsenkurs Vorhersage **  
**Using ML to predict aktien kurse. Anzapfen von aktien kursen via einer API, ETL prozesse und speichern von raw und veränderten daten. Training eines RNN um eine zeitreihen analyse durch zu führen und eine prediciton ab zu geben. Evaluation im nachgang: 'Was wäre passiert wenn ich investiert hätte' und bei einem Schwellen wert wieder verkauft hätte. Zu guter letzte eine automatisierung des scripts mittels eines shell-scripts / cronjobs welches automatische in regelmäßigen abständen den Prozess ausführt.**

==============================

## **Table of Contents**  
1. [Overview](#overview)  
2. [Project Structure](#project-structure)  
3. [Dataset] usw 

==============================

## **1. Overview**  


==============================


## **2. Project Structure**
------------
stock-rnn-prediction/
    ├── data/                               # Roh- und Verarbeitete Daten
    │   ├── raw/                            # Unveränderte API-Daten
    │   ├── processed/                      # Bereinigte, normalisierte Daten
    │   └── predictions/                    # Modellvorhersagen
    │
    ├── notebooks/                          # Explorative Jupyter Notebooks
    │   ├── collect_data_api.ipynb          # Notebook welches die daten von der API zieht 
    │   └── analysis.ipynb
    │
    ├── models/                   # Gespeicherte RNN-Modelle
    │   └── rnn_model.h5
    │
    ├── reports/                  # Ergebnisse, Visualisierungen
    │   └── trades_summary.csv
    │
    ├── scripts/                  # Shell- und Automatisierungsskripte
    │   └── update_data.sh
    │
    ├── src/                      # Quellcode
    │   ├── api/                  # API-Zugriff
    │   │   └── fetch_data.py
    │   │
    │   ├── etl/                  # Extraktion, Transformation, Laden
    │   │   ├── transform.py
    │   │   └── load.py
    │   │
    │   ├── model/                # RNN-Modelle und Training
    │   │   ├── rnn_model.py
    │   │   └── train.py
    │   │
    │   ├── utils/                # Hilfsfunktionen, Logging, etc.
    │   │   └── helpers.py
    │   │
    │   ├── evaluation/           # Auswertung: Buy/Sell-Strategie
    │   │   └── strategy_eval.py
    │   │
    │   └── main.py               # Einstiegspunkt, orchestriert alles
    │
    ├── requirements.txt          # Abhängigkeiten
    ├── README.md                 # Projektbeschreibung
    ├── .env                      # Konfigurationsdateien
    └── .gitignore

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



### Model Analysis



**Recommendation**: XGBoost 

### Outlook and Improvements



==============================

## **7. Contributing**

This project was solemnly set up, idea and finalizes by me and myselfe. Got only help by perplexity and chaaaat. 

==============================

## **8. License**

This project is licensed under the MIT License.