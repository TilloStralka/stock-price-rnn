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

│
├── data/                             # All data used in the project
│   ├── raw/                          # Untouched API data as CSV files
│   ├── processed/                    # Cleaned and normalized data
│   └── predictions/                  # Model-generated prediction results
│
├── notebooks/                        # Jupyter notebooks for exploration and prototyping
│   ├── collect_data_api.ipynb        # Notebook for fetching stock data via API
│   └── analysis.ipynb                # Planned notebook for EDA on raw CSVs
│
├── models/                           # Saved model weights and architectures
│   └── rnn_model.h5                  # Trained RNN model file
│
├── reports/                          # Output results and visualizations
│   └── trades_summary.csv            # Summary of backtested trades
│
├── scripts/                          # Shell scripts and ETL pipeline triggers
│   ├── run_daily_etl.sh             # Daily cron-triggered ETL runner
│   ├── run_etl.sh                   # Planned general ETL shell runner
│   ├── run_extract.py               # Selects stocks and fetches raw data
│   └── run_transform.py             # Transforms raw CSVs for RNN input
│
├── src/                              # Project source code
│   ├── model/                        # RNN model definition and training
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   ├── evaluation/                   # Buy/sell strategy evaluation
│   │   └── strategy_eval.py
│   │
│   └── utils.py                      # Utility functions (e.g., logging, configs)
│
├── logs/                             # Cronjob and script output logs
│   └── cron_etl.log
│
├── test/                             # Unit tests for individual components
│   └── test_utils.py
│
├── requirements.txt                  # Python package dependencies
├── README.md                         # Project overview and instructions
├── .env                              # Environment-specific settings (e.g. conda env, data paths)
└── .gitignore                        # Files and folders to ignore in Git

```

==============================

## **3. Dataset**
The data comes from the api ... bla bla bla doku füllen . [Source](https://www.fill in here)

==============================

## **4. Installation**
1. Clone the repository:
    ```bash
    git clone https://github.com/TilloStralka/stock-price-rnn.git
    cd stock-price-rnn

    ```

2. Create a virtual environment and activate it:
    ```bash
    conda create -n stock_rnn_env python=3.10
    conda activate stock_rnn_env
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    
4. Conda shell integration (if not yet enabled)
    ```bash
    conda init zsh # or: conda init bash
    ```

5. Make the daily ETL shell script executable

    ```bash 
    chmod +x scripts/run_daily_etl.sh
    ```
6. Configure .env
    Create a .env file with your environment settings:

    CONDA_ENV_NAME=stock_rnn_env
    CONDA_PYTHON_PATH=/opt/anaconda3/envs/stock_rnn_env/bin/python
    PROJECT_PATH=/Users/yourname/path/to/stock-price-rnn


7. Daily Automation via Cron
    To run the ETL pipeline every day at 8:00 AM, follow these steps:

    Open your crontab:
    ```bash
    crontab -e
    ````
    Add the following line (adjust the path as needed):
    ```bash 
    0 8 * * * /bin/bash /Users/yourname/path/to/stock-price-rnn/scripts/run_daily_etl.sh >> /Users/yourname/path/to/stock-price-rnn/logs/cron_etl.log 2>&1
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