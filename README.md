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
```text
------------
stock-rnn-prediction/
    ├── data/                               # Roh- und Verarbeitete Daten
    │   ├── raw/                            # Unveränderte API-Daten als CSV
    │   ├── processed/                      # Bereinigte, normalisierte Daten
    │   └── predictions/                    # Modellvorhersagen
    │
    ├── notebooks/                          # Explorative Jupyter Notebooks
    │   ├── collect_data_api.ipynb          # Notebook welches die daten von der API zieht 
    │   └── analysis.ipynb                  # Geplantes exploratives notebook zur analyse der raw csv
    │
    ├── models/                   # Gespeicherte RNN-Modelle
    │   └── rnn_model.h5
    │
    ├── reports/                  # Ergebnisse, Visualisierungen
    │   └── trades_summary.csv
    │
    ├── scripts/                  # Shell- und Automatisierungsskripte
    │   └── update_data.sh          # Update der daten, shell script, ruft automatisch run_extract usw auf 
    │   └── run_extract.py          # Hier werden die Aktien definiert welche geladen werden 
    │   └── run_etl.sh          # geplantes Shell-skript das ETL System periodisch auf zu rufen 
    │
    ├── src/                      # Quellcode  // Geplante Struktur, better all in 1 utils.py function?
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
    ├── test/                     # Test functionen der einzelnen functionen
    │   └── test_utils.py
    │
    ├── requirements.txt          # Abhängigkeiten
    ├── README.md                 # Projektbeschreibung
    ├── .env                      # Konfigurationsdateien
    └── .gitignore
```

==============================

## **3. Dataset**
The data comes from the api ... bla bla bla doku füllen . [Source](https://www.fill in here)

==============================

## **4. Installation**
1. Clone the repository:
    ```bash
    git clone https://https://github.com/TilloStralka/stock-price-rnn.git
    cd stock-price-rnn
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
bla bla will be downloaded into `data/raw/` folder. Ensure the file is named as required for seamless processing.

##### **5.1.2. Process the Data**  


##### **5.1.3. Finalized Dataset**  


#### **5.2. Modeling**
To perform the modeling, use the notebook:  
- Notebook: `ModelPipeline.ipynb`  
- This notebook contains all the necessary steps, fully commented for clarity.  

#### **5.3. Utilities**
The functions called in the notebooks are implemented in the `utils.py` file. You can review this file for additional details about the underlying implementation.

#### **5.4. Trained Models**
After training, the trained models are saved in the `models/` folder for future use.  

---

#### **5.5. Additional Notes**
For any modifications to the processing or modeling pipeline, ensure that changes are consistent across all related files. For further assistance, consult the documentation and comments within the notebooks.

==============================

## **6. Results**



### Model Analysis


### Outlook and Improvements



==============================

## **7. Contributing**

This project was solemnly set up, idea and finalizes by me and myselfe. Got only help by perplexity and chaaaat. 

==============================

## **8. License**

This project is licensed under the MIT License.