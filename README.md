# Spotify Track Popularity Prediction Using Machine Learning

## Overview

This project investigates the factors that influence Spotify track popularity by developing and comparing multiple machine learning models. The objective is to determine whether a song's popularity is driven primarily by its audio characteristics or by the popularity of its artist.

Three predictive models were implemented and evaluated using standard regression metrics. Model interpretability techniques were also applied to better understand feature importance and prediction behaviour.

---

## Objectives

- Predict Spotify track popularity using machine learning models
- Compare the performance of different predictive algorithms
- Evaluate models using R² and Mean Absolute Error (MAE)
- Interpret feature importance using SHAP analysis
- Investigate the relative influence of artist popularity and audio features

---

## Dataset

The dataset contains Spotify tracks together with:

- Audio characteristics (danceability, energy, valence, tempo, etc.)
- Artist popularity
- Monthly listeners
- Track popularity score

---

## Methodology

The project followed the following workflow:

1. Data cleaning and preprocessing
2. Feature engineering
3. Model development
4. Model evaluation
5. Error analysis
6. Model interpretation using SHAP

---

## Machine Learning Models

The following models were implemented and compared:

- Linear Regression
- Random Forest
- Gradient Boosting

---

## Evaluation Metrics

The models were evaluated using:

- R² Score
- Mean Absolute Error (MAE)

Random Forest achieved the strongest predictive performance among the three models.

---

## Key Findings

- Artist popularity is a significantly stronger predictor of track popularity than audio characteristics alone.
- Random Forest outperformed both Linear Regression and Gradient Boosting.
- SHAP analysis showed that market-related variables (such as monthly listeners and artist popularity) contributed more to prediction performance than acoustic features.
- The models tended to overestimate low-popularity tracks and underestimate highly popular tracks, demonstrating a common regression-to-the-mean effect.

---

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- SHAP
- Matplotlib
- Jupyter Notebook

---

## Repository Structure

```
spotify-track-popularity-prediction/
│
├── data/                
├── notebooks/            # Jupyter notebooks
├── report/               # Project report
├── README.md
└── Spotify Prediction.ipynb (or .py)
```

---

## Results

| Model | R² (Without Artist Features) | R² (With Artist Features) |
|-------|-----------------------------:|--------------------------:|
| Linear Regression | 0.04 | 0.44 |
| Random Forest | 0.693 | 0.873 |
| Gradient Boosting | 0.084 | 0.53 |

Including artist-related features substantially improved prediction performance across all models.

| Model | MAE (Without Artist Features) | MAE (With Artist Features) |
|-------|------------------------------:|---------------------------:|
| Linear Regression | 14.23 | 9.91 |
| Random Forest | 6.237 | 4.231 |
| Gradient Boosting | 13.57 | 8.59 |

## Future Improvements

Potential future improvements include:

- Incorporating temporal features to capture changes in popularity over time
- Testing additional machine learning algorithms (e.g. XGBoost)
- Hyperparameter optimisation
- Incorporating Spotify API data for real-time prediction
