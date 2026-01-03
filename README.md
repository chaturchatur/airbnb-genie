# Airbnb gennie

Project for analyzing airbnb listings and predicting prices using machine learning.

## Project structure

- `data/`: contains raw and processed data files
- `models/`: directory for saved model artifacts
- `results/`: output plots and analysis figures
- `src/`: source code for data processing and modeling
  - `clean_data.py`: cleaning and feature engineering pipeline
  - `exploratory_analysis.py`: exploratory data analysis and visualization
  - `baseline_model.py`: baseline linear regression and random forest models
  - `baseline_model_v2.py`: improved histogram-based gradient boosting model with hyperparameter tuning
- `Dockerfile`: docker configuration for containerization
- `requirements.txt`: python dependencies

## Setup

### Local installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Docker

Build the image:

```bash
docker build -t airbnb-gennie .
```

Run the container:

```bash
docker run -p 8501:8501 airbnb-gennie
```

## Usage

Run the scripts in the following order to reproduce the analysis:

1. **Data cleaning**:
   Reads raw data, handles missing values, and creates features.

   ```bash
   python src/clean_data.py
   ```

2. **Exploratory analysis**:
   Generates distributions, correlation heatmaps, and spatial plots in `results/`.

   ```bash
   python src/exploratory_analysis.py
   ```

3. **Baseline modeling**:
   Trains linear regression and random forest models, evaluating with rmse/mae/r2.

   ```bash
   python src/baseline_model.py
   ```

4. **Improved modeling**:
   Trains a tuned gradient boosting regressor and saves the best model.
   ```bash
   python src/baseline_model_v2.py
   ```

## Results

Key outputs are saved in the `results/` directory, including:

- Actual vs predicted price plots
- Feature importance charts
- Correlation heatmaps
- Spatial distribution maps
