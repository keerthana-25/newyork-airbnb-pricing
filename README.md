# CRISP-DM — New York Airbnb Pricing
## 1. Business Understanding

Objective: predict nightly price for NYC Airbnb listings so hosts can set competitive rates and analysts can flag outliers.

Primary metric: MAE in dollars. Also track RMSE and R².

Baseline: median price by neighborhood × room_type. A model is “shippable” only if it beats this baseline by a meaningful margin.

## 2. Data Understanding

Assets: datasets/ with listings (and calendar if available), plus exploratory and modeling notebooks under notebooks/.

Key fields: neighborhood, latitude/longitude, room_type, accommodates, beds/bedrooms/bathrooms, minimum_nights, availability, review_scores*, number_of_reviews, amenities text, target price.

Initial findings: price is heavy-tailed; neighborhood and room_type explain a lot of variance; availability and review signals correlate with price; text features (amenities/name) are noisy but useful.

## 3. Data Preparation

Cleaning: strip currency symbols → numeric; standardize bath/bed dtypes; drop duplicates; sanity-check coordinates.

Missing values: median (numeric) and mode (categorical) imputation within neighborhood × room_type; drop rows missing target.

Outliers: winsorize price at 1st/99th pct; remove zero-accommodates or extreme minimum_nights.

Feature engineering:

Model on log_price; back-transform to dollars for reporting.

Geo: haversine distance to city center/POIs from lat/lon.

Host: tenure from host_since, superhost flag, review density.

Demand: calendar aggregates (weekend vs weekday availability, booked ratio) if calendar data exists.

Text: top-k amenities one-hots (wifi, AC, washer, parking).

Encoding: one-hot for low-cardinality; mean/target encoding with CV for high-cardinality (neighborhood).

Scaling: standardize numeric features for linear models; tree models use raw scales.

Splits: train/valid/test stratified by neighborhood and room_type; use time-aware split if calendar features could leak.

## 4. Modeling

Tried: Linear/L2, Lasso, Elastic Net; Random Forest; Gradient Boosting/XGBoost; CatBoost (for categorical handling).

Protocol: sklearn Pipeline/ColumnTransformer bundles preprocessing + model to avoid leakage; 5-fold CV; randomized/Bayesian search; early stopping for boosters.

Pick: simplest model within ~1–2% of best validation MAE.

## 5. Evaluation

Report: CV and hold-out Test MAE/RMSE vs baseline.

Calibration: residuals vs fitted and by price deciles; check under/over-pricing at extremes.

Slices: MAE by neighborhood, room_type, host tenure, review buckets.

Explainability: permutation importance/SHAP to surface drivers (location, room_type, accommodates, amenities).

Error review: inspect highest-error listings; adjust caps/encodings/features if patterns repeat.

## 6. Deployment & Maintenance

Artifact: versioned joblib/pickle of the full sklearn pipeline + model_card.md (feature ranges, intended use, caveats).

Serving (optional): FastAPI POST /predict returns predicted price and a simple CI (use CV residual std).

Monitoring: validate schema/ranges, track feature drift vs training snapshot, and MAE on any observed ground truth; trigger retrain on drift or schedule quarterly.


## Links:
Dataset : https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data

Chat GPT : https://chatgpt.com/c/68d48835-f0e4-8325-a5f3-e9707ddfcc07

Medium : https://medium.com/@keerthanapm257/the-crisp-dm-methodology-in-data-science-e1caa43d60af
