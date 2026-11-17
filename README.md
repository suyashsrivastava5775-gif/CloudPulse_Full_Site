# CloudPulse — Full Cloud Resource Prediction Website

A complete local web application for the research project **Cloud Resource Usage Prediction Using a Lightweight Machine Learning Model**.

## Requirements
- Python 3.10 or newer
- Internet connection for Google Fonts and Chart.js CDN (the dashboard itself runs locally; charts need the CDN)

## Windows setup
1. Extract `CloudPulse_Full_Site.zip`.
2. Open the extracted `CloudPulse_Full_Site` folder in VS Code.
3. Open Terminal in that folder.
4. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate
```

5. Install packages:

```bash
pip install -r requirements.txt
```

6. Start the website:

```bash
python app.py
```

7. Open this address in your browser:

`http://127.0.0.1:5000`

Stop the server with `Ctrl + C`.

## Features
- Flask backend and responsive dashboard
- CPU, memory, network and disk usage metrics
- CPU and memory trend chart
- Random Forest regression using scikit-learn
- Chronological 80/20 train-test split
- MAE, RMSE and R² model evaluation
- Next-interval CPU and memory forecast
- Low / medium / high capacity risk recommendation
- Manual resource-reading entry
- CSV dataset upload and export
- Downloadable sample CSV
- Reset to reproducible synthetic demo data

## CSV format
The uploaded CSV must have these exact columns:

`timestamp,cpu,memory,network,disk,workloads`

Example:
`2026-11-17 10:00:00,45,62,280,34,110`

- `timestamp`: date/time parseable by pandas
- `cpu`, `memory`: percentages from 0 to 100
- `network`: Mbps
- `disk`: disk I/O in MB/s
- `workloads`: active workload count
- At least 20 valid rows are required.

## Research methodology
1. Collect timestamped cloud workload observations.
2. Validate, clean, and chronologically sort the dataset.
3. Generate time-of-day sine/cosine features.
4. Use current interval CPU, memory, network, disk and workload count to predict CPU and memory in the next interval.
5. Train a Random Forest regression model with 140 estimators.
6. Evaluate on a chronological 80/20 holdout set using MAE, RMSE and R².
7. Convert predictions into a capacity-risk recommendation.

## Important academic note
The initial dataset is synthetic demonstration data generated locally. Do not claim it represents a real cloud provider or use its metrics as real experimental findings. For the final research evaluation, upload a documented real workload trace and report the resulting metrics. The application is not connected to AWS, Azure or GCP and does not provision cloud resources.
