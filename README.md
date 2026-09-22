#Goa Beach Waste Prediction

Daily conditions in. A practical waste estimate out. Field observations stay central.

Goa Beach Waste Prediction is a machine-learning case study for estimating daily beach-waste generation in Goa, India. It combines an actual daily waste target with environmental, calendar, tourism, and demographic information, then exposes the trained model through an interactive Streamlit dashboard.

The project was partly inspired by Apte et al.'s 2023 Mumbai beach-waste study, but it is deliberately a Goa-specific case study. It asks a simple operational question: given the calendar and observed or expected conditions around a day, what daily beach-waste quantity does the model estimate in metric tonnes?

Project scope: Goa, India · daily observations from 2022–2025 · regression target: beach_waste_daily_mt

The problem

Beach-waste planning needs timely, local information. Collection teams have to decide when routine operations may be enough and when weather, seasonality, visitor activity, or events may justify closer monitoring. Daily waste generation is variable, however, and clean daily target data is not easy to find.

One of the hardest parts of this project was not selecting a model. It was finding a defensible target variable.

During the early work, I searched public reports, economic surveys, waste-management records, and government documents. Some sources contained useful overall or cumulative waste figures, but not the daily time series needed for a forecasting model. Other documents were unavailable or unsuitable for extracting a daily target. I did not want to turn aggregate figures into a made-up daily series just to make the modelling easier.

The breakthrough came through direct communication with an Information Officer at the Goa Waste Management Corporation (GWMC), which made a daily beach-waste dataset available for 2022–2025. That is why this project uses an actual daily waste target rather than a synthetic estimate.

The idea

The model estimates daily beach waste from a compact set of temporal, environmental, tourism, and demographic inputs. It is not intended to replace field measurements or dictate collection decisions. Instead, it gives researchers and operational teams a reproducible way to explore the patterns available in the study data and test scenario inputs.

The dashboard accepts conditions for a selected date, derives calendar features such as weekend and monsoon status, and returns a predicted quantity in metric tonnes. The historical explorer keeps the model grounded by allowing users to inspect the observations behind the work.

The central claim is modest: the available features carry useful predictive signal for this dataset, while a meaningful share of day-to-day variation remains unexplained.

Three parts of the project

Daily target data — beach_waste_daily_mt, provided through GWMC communication and recorded as a daily quantity in metric tonnes. This is the outcome the model learns to estimate.

Feature pipeline — calendar, public-holiday and event indicators; IMD rainfall; Copernicus Marine wave conditions; wind speed; Goa population estimates; and annual tourism totals. Some variables are derived from dates or represented at annual rather than daily frequency.

Forecasting and interpretation — chronologically evaluated regression models, time-aware hyperparameter tuning, SHAP feature-importance analysis, and a Streamlit interface for scenario exploration.

Results

The final reported model is a tuned Gradient Boosting Regressor, evaluated on the held-out 2025 period. The earlier 2022–2024 observations were used for training.

Held-out 2025 metric

Result

MAE

0.6603 MT/day

RMSE

0.8879 MT/day

R²

0.5485

The result indicates meaningful predictive signal, not a complete explanation of daily beach-waste variation. It should be read alongside the study period, the available features, and the limitations described below.

Baseline models

Model

Train MAE

Train RMSE

Train R²

Test MAE

Test RMSE

Test R²

Linear Regression

0.7073

0.9684

0.3956

4.1396

4.2649

-9.4168

Decision Tree

0.0000

0.0000

1.0000

0.8314

1.1274

0.2721

Random Forest

0.1748

0.2576

0.9572

0.6461

0.8722

0.5643

Gradient Boosting

0.3786

0.5348

0.8157

0.6603

0.8879

0.5485

Tuned tree-based models

Hyperparameter search used GridSearchCV with TimeSeriesSplit(n_splits=5), rather than random cross-validation.

Model

Best CV R²

Held-out 2025 MAE

Held-out 2025 RMSE

Held-out 2025 R²

Decision Tree

0.1233

0.6250

0.8911

0.5452

Random Forest

0.2815

0.6682

0.8943

0.5420

Gradient Boosting

0.2884

0.6603

0.8879

0.5485

The selected Gradient Boosting configuration was:

{
    "learning_rate": 0.1,
    "max_depth": 3,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "n_estimators": 100,
}

How to read these numbers

Three details matter more than a single headline metric:

The Decision Tree's perfect training score is a warning sign, not a success claim. Its much weaker held-out result shows clear overfitting.

Linear Regression generalized very poorly to 2025. The tree-based approaches performed materially better on the chronological holdout.

The best tuned test R² is about 0.55. That is evidence that the supplied features contain useful information, but it also means substantial daily variation is not captured by the current model.

The MAE of 0.6603 MT/day is easiest to read as the average absolute error scale on the held-out 2025 data. It is not a guarantee for every future day or every beach in Goa.

Architecture

GWMC daily beach-waste target ──┐
IMD gridded rainfall ───────────┤
Copernicus wave reanalysis ─────┤
Wind, population, tourism ──────┤──> cleaned daily dataset
Calendar, holidays, events ─────┘             │
                                               │ feature engineering
                                               v
                                chronological train / test split
                                2022–2024 train · 2025 test
                                               │
                                               v
                         baseline models → time-aware tuning → Gradient Boosting
                                               │
                              ┌────────────────┴────────────────┐
                              v                                 v
                 SHAP global feature importance          Streamlit dashboard
                                                        scenario forecast + EDA

The dashboard loads the saved model and its feature schema, constructs a model-ready row in the expected order, and returns the predicted daily amount. The app also displays the historical data and descriptive views; it does not retrain the model on each visit.

Quickstart

Clone or download the repository, then create an isolated Python environment:

git clone <your-repository-url>
cd Goa_Coastal_Waste

python -m venv .venv

Activate it:

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

Install the dependencies and start the dashboard:

pip install -r requirements.txt
streamlit run app.py

Then open the local URL printed by Streamlit, usually http://localhost:8501.

The serialized model was created with scikit-learn 1.6.1. The repository pins that version in requirements.txt so that the model can be loaded consistently.

The dashboard

The Streamlit application has three sections:

Forecast workspace — select a date and set environmental or activity assumptions. Weekend and monsoon features are calculated from the date; the model estimates daily waste in metric tonnes.

Evidence explorer — filter the 2022–2025 observations, inspect daily trends, monthly and day-of-week summaries, simple associations, and download the filtered records as CSV.

Data & methodology — a concise record of the study period, modelling approach, feature families, and responsible-use limits.

Live Demo Link · Demo Video

These links are placeholders until a live Streamlit URL or video is available.

Dataset

The final dataset contains 1,461 daily observations, from 01 January 2022 through 31 December 2025. There were no duplicate rows and no missing target values.

Summary statistic for beach_waste_daily_mt

Metric tonnes/day

Count

1,461

Mean

3.030527

Standard deviation

1.266281

Minimum

0.631

25th percentile

2.105

Median

2.800

75th percentile

3.661

Maximum

11.164

Two data-quality issues were handled before modelling:

Missing holiday_name values were recorded as “No Holiday.”

One missing wave_height_m value was filled with the mean wave height.

The raw target did not have missing values.

Data sources and inputs

Source or input family

Role in the project

Goa Waste Management Corporation (GWMC)

Daily beach-waste target

India Meteorological Department (IMD)

Daily gridded rainfall

Copernicus Marine global ocean waves reanalysis

Significant wave height (VHM0)

Wind data

wind_speed_10m_mps

Goa population estimates

population_thousand

Goa tourism statistics

Annual domestic, foreign, and total tourist counts

Goa holiday and calendar information

Calendar and public-holiday indicators

Goa festival/event information

Major-event and lagged-event indicators

For rainfall, the IMD 0.25° × 0.25° daily grid was spatially summarized over an approximate Goa-area bounding box, using the spatial mean as the daily rainfall value. This is not presented as an exact administrative-boundary mask.

Feature engineering

The original date field was converted into numerical and indicator features, then removed before modelling. The final model uses 18 predictors:

Feature family

Included predictors

Calendar

day, month, year, day_of_week_num, is_weekend, is_monsoon, is_public_holiday, is_major_festival_or_event

Environmental

rainfall_mm, rainfall_lag1, rainfall_3day_sum, wave_height_m, wind_speed_10m_mps

Demographic and activity proxy

population_thousand, total_tourists_annual

Lagged calendar/event context

is_weekend_lag1, is_public_holiday_lag1, is_major_festival_or_event_lag1

The exact model input order is saved in Model/feature_names.json. Not every predictor is an independent daily measurement: some values are derived from the calendar and tourism is represented by annual totals.

Exploratory data analysis

The EDA covered target distribution, daily trends, monthly and yearly summaries, day-of-week behaviour, lagged weekend/event comparisons, correlations, and heatmaps.

One result that surprised me was the seasonal pattern. I initially expected the highest levels to be concentrated in summer. In this dataset, waste was relatively higher around the post-monsoon and winter period: December had the highest monthly average in the analysis, while June–September were generally lower. That observation led to a more careful interpretation of seasonality and beach activity rather than an assumption based only on summer tourism.

Sunday showed a high average waste level, and the analysis suggested that weekend effects could continue into the following day. This motivated the use of is_weekend_lag1.

Rainfall was examined as both an immediate and delayed condition through rainfall_mm, rainfall_lag1, and rainfall_3day_sum. The later SHAP analysis found the three-day accumulated-rainfall feature to be more influential to this model than same-day rainfall. That describes model behaviour, not a causal conclusion.

Modelling approach

Beach-waste prediction is treated as a regression task because the target is continuous. The first comparison included:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Gradient Boosting Regressor

Each model was evaluated with MAE, RMSE, and R². Tree-based models were subsequently tuned with a time-aware cross-validation setup.

Time-series train/test strategy

This project does not use a random train_test_split.

2022–2024  →  training set  (1,096 observations)
2025       →  test set      (365 observations)

The aim is to mimic the real forecasting direction: learn from earlier periods and evaluate against a later year. Randomly mixing dates across 2022–2025 could let information from future periods enter the training data and would give a less realistic assessment.

Final model

The final deployed artifact is the tuned Gradient Boosting Regressor, retained with total_tourists_annual included. It is saved at:

Model/goa_beach_waste_model.pkl

Its companion file, Model/feature_names.json, preserves the expected 18-feature input order. The final result reported for the held-out 2025 period is MAE = 0.6603 MT/day, RMSE = 0.8879 MT/day, and R² = 0.5485.

SHAP explainability

SHAP was used to inspect the final Gradient Boosting model's global feature reliance. The values below are mean absolute SHAP importance values from the project analysis.

Feature

SHAP importance

month

0.491809

rainfall_3day_sum

0.339939

total_tourists_annual

0.119634

population_thousand

0.101372

wave_height_m

0.093853

is_monsoon

0.082275

rainfall_lag1

0.070390

day_of_week_num

0.061217

year

0.056901

day

0.046755

is_major_festival_or_event

0.040571

is_weekend

0.032935

wind_speed_10m_mps

0.030544

rainfall_mm

0.023965

is_major_festival_or_event_lag1

0.017743

is_weekend_lag1

0.011437

is_public_holiday

0.006087

is_public_holiday_lag1

0.000024

Seasonality through month and accumulated recent rainfall through rainfall_3day_sum were especially influential in this model. Public-holiday lag made very little contribution. SHAP shows how the fitted model uses its inputs; it does not establish that any input causes beach waste to change.

Key findings

Obtaining a real daily target was a central part of the work. Direct institutional communication made it possible to avoid fabricating a daily series from unrelated aggregates.

A chronological evaluation was important. The model was trained on 2022–2024 and tested on 2025 rather than on randomly mixed dates.

The baseline Decision Tree overfit strongly, while Random Forest and Gradient Boosting generalized more credibly to the held-out year.

The final model captures meaningful predictive signal, but the held-out R² of about 0.55 also leaves substantial variation unexplained.

The monthly pattern was not simply a summer peak; the EDA highlighted relatively higher post-monsoon and winter averages, including December.

The model relied more on recent accumulated rainfall than on same-day rainfall in the SHAP analysis.

Scope and limitations

This is a Goa-specific study, not a universal model for every beach or coastline.

Study design

The data covers only 2022–2025, and the final future holdout is one year: 2025.

Model performance must be interpreted within this period and forecasting setup.

The application is decision support, not a substitute for field measurements, local knowledge, or collection logs.

Inputs

Some variables are annual or derived rather than direct daily observations. Tourism is represented by annual totals.

IMD rainfall was aggregated with an approximate Goa-area bounding box, not an exact administrative mask.

Beach-specific visitor counts, cleaning schedules, local events, coastal currents, flooding, waste-source composition, and other local conditions are not directly captured. These are plausible missing inputs, not demonstrated explanations for individual prediction errors.

Interpretation

The model does not explain all day-to-day variation in waste.

SHAP explains fitted model behaviour, not causal relationships.

A forecast for conditions far outside the ranges observed in 2022–2025 should be treated cautiously.

What I learned during the project

The data-acquisition stage changed the project. At first, publicly available aggregate reports seemed like a possible route, but they could not support a defensible daily prediction target. Rather than filling that gap with an estimated series, I continued looking for the underlying data and reached out directly. That decision made the project slower to start, but much stronger as a real-world case study.

The modelling stage was equally instructive. The perfect training score of the untuned Decision Tree looked impressive at first glance, but the holdout result made the overfitting obvious. The more useful lesson was that a model has to be evaluated in the direction it will actually be used: from past dates to future dates.

Repository structure

Goa_Coastal_Waste/
├── app.py                                  # Streamlit dashboard
├── Goa_Beach.csv                           # Daily study dataset
├── requirements.txt                        # Python dependencies
├── README.md
├── .gitignore
├── Model/
│   ├── goa_beach_waste_model.pkl           # Trained Gradient Boosting model
│   └── feature_names.json                  # Required predictor order
└── Notebook/
    └── Coastal.ipynb                       # EDA and modelling workflow

Deployment

The project is designed to run as a Streamlit app. To deploy it with Streamlit Community Cloud or another Python host, include:

app.py
Goa_Beach.csv
requirements.txt
Model/goa_beach_waste_model.pkl
Model/feature_names.json

Set app.py as the Streamlit entry point. No secrets are required for the included dashboard. The .gitignore intentionally excludes local secret files such as .streamlit/secrets.toml while keeping the data and trained model available for deployment.

Future improvements

Extend the target series over additional years and evaluate across more future holdout periods.

Incorporate beach-specific observations where they can be obtained responsibly.

Add daily visitor or footfall measures instead of relying only on annual tourism totals.

Evaluate localized hydrodynamic, cleaning-schedule, flooding, and waste-composition variables where data is available.

Compare further time-series models and quantify prediction uncertainty.

Add automated data validation and reproducible pipeline scripts as the data collection workflow matures.

Related research

This project was partly inspired by:

Apte et al. (2023), Machine learning approach for automated beach waste prediction and management system: A case study of Mumbai, Frontiers in Mechanical Engineering. https://doi.org/10.3389/fmech.2023.1120042

The Mumbai study used daily waste data from Dadar-Mahim beach from 2013–2018 and also followed a chronological framing, training on earlier years and testing on the final year. The Goa project follows a similar broad problem framing while using Goa-specific target data, local environmental and calendar inputs, engineered lagged features, time-series cross-validation, hyperparameter tuning, and SHAP interpretation.

The Mumbai paper reports average validation “accuracy” values of 52.7% for Random Forest and 58.41% for Linear Regression. Those figures are not presented here as R² values: the paper does not clearly define them as standard regression metrics and does not report MAE or RMSE for those prediction models. Its separately reported R² values for yearly population-related analyses are different analyses and should not be compared directly with this project's held-out prediction R².

Acknowledgements

Thank you to the Goa Waste Management Corporation and the Information Officer whose communication made access to a daily beach-waste target possible. The project also draws on IMD rainfall data, Copernicus Marine wave reanalysis, Goa population and tourism information, and public calendar and event information.

Built with

Python · pandas · scikit-learn · SHAP · Streamlit · Plotly

The project is documented as a research-oriented learning and forecasting case study. The model, assumptions, exploratory findings, and limitations are kept visible so that another researcher, student, recruiter, or stakeholder can see what was actually built and where the work can improve.
