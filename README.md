# XAI\_2026 — Speech Emotion Recognition with Explainable AI

## Team Members & Responsibilities

|Name|Student ID|Responsibilities|
|-|-|-|
|Kacper Geisshirt|443171| virtual enviroment, data preprocessing |
|Shagufta Shaheen|477654| modelling |

\---

## Overview

This project builds a Speech Emotion Recognition (SER) system that takes short voice clips from the RAVDESS dataset and predicts the speaker's emotion using audio features. The project goes beyond standard model training by applying Explainable AI (XAI) methods to interpret and validate model decisions.

Two models are trained and explained:

* **XGBoost** on engineered MFCC features, explained using **SHAP**
* **CNN + BiLSTM** on MFCC sequences, explained using **Grad-CAM**

The central research question is whether two independent XAI methods applied to two different models arrive at the same acoustic conclusions about emotion in speech.

\---

## Dataset

**Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**

* Source: https://zenodo.org/records/1188976
* This project uses the speech-only audio subset
* 1440 files total: 60 trials x 24 actors
* 24 professional actors (12 male, 12 female)
* North American neutral accent
* 8 emotion classes: neutral, calm, happy, sad, angry, fearful, disgust, surprised
* Two intensity levels: normal and strong
* Sample rate: 48,000 Hz

### File Naming Convention

Each filename follows a 7-part identifier system:

```
03-01-06-01-02-01-12.wav
 |   |   |   |   |   |   |
 |   |   |   |   |   |   Actor (12 = female)
 |   |   |   |   |   Repetition (01 = first)
 |   |   |   |   Statement (02 = dogs)
 |   |   |   Intensity (01 = normal)
 |   |   Emotion (06 = fearful)
 |   Vocal channel (01 = speech)
 Modality (03 = audio only)
```

### Emotion Label Mapping

|Code|Emotion|
|-|-|
|01|Neutral|
|02|Calm|
|03|Happy|
|04|Sad|
|05|Angry|
|06|Fearful|
|07|Disgust|
|08|Surprised|

\---

## Project Structure

```
Project/                                          <- Root (Shagufta Shaheen/4XAI/Project)
│
├── .git/
├── .gitignore
├── README.md                                     <- This file
│
├── data/
│   ├── raw/                                      <- Original .wav files (not tracked in Git)
│   │   └── Actor\_01/ ... Actor\_24/               <- Download from Zenodo
│   │
│   └── processed/                                <- Preprocessed feature files
│       ├── ravdess\_metadata.csv                  <- File paths and emotion labels (1440 rows)
│       ├── ravdess\_mfcc\_features.csv             <- Original 40 mean MFCC features
│       ├── ravdess\_mfcc\_features\_v2.csv          <- Extended 160 MFCC features
│       ├── ravdess\_emotion\_labels\_encoded.npy    <- Numeric emotion labels (1440,)
│       └── ravdess\_emotion\_classes.npy           <- Class name lookup array (8,)
│
└── XAI\_2026/
    │
    ├── README.md                                 <- Project README (inside XAI folder)
    ├── pyproject.toml                            <- Python project config
    ├── .python-version                           <- Python version spec
    ├── uv.lock                                   <- Dependency lock file
    │
    ├── notebooks/
    │   ├── Preprocessing.ipynb                   <- Original preprocessing (Kacper)
    │   │                                            Extracts labels, standardizes audio,
    │   │                                            saves 40 MFCC features + mel spectrograms
    │   │
    │   ├── 01\_preprocessing\_v2.ipynb             <- Extended preprocessing (Shagufta)
    │   │                                            Adds std, delta, delta\_std features
    │   │                                            Expands MFCC from 40 to 160 features
    │   │                                            Saves ravdess\_mfcc\_features\_v2.csv
    │   │
    │   ├── 02\_classical\_ml\_shap.ipynb            <- Phase 1: XGBoost + SHAP (Kacper)
    │   │                                            Feature scaling and train/test split
    │   │                                            Random Forest and XGBoost training
    │   │                                            RandomizedSearchCV hyperparameter tuning
    │   │                                            Confusion matrix and per-emotion F1
    │   │                                            SHAP global, dot, per-emotion,
    │   │                                            waterfall and heatmap analysis
    │   │
    │   ├── 03\_cnn\_gradcam.ipynb                  <- Phase 2: CNN+BiLSTM + Grad-CAM (Shagufta)
    │   │                                            CNN+BiLSTM model training on MFCC sequences
    │   │                                            Enhanced early stopping
    │   │                                            Confusion matrix and classification report
    │   │                                            Grad-CAM implementation for 1D CNN
    │   │                                            Per-emotion Grad-CAM heatmaps
    │   │                                            Misclassification Grad-CAM analysis
    │   │
    │   └── 04\_cnn\_bilstm\_gradcam\_comparison\_conclusion.ipynb
    │                                             <- Phase 3: Comparison + Conclusion (Shagufta)
    │                                                Performance comparison XGBoost vs CNN+BiLSTM
    │                                                Per-emotion F1 comparison
    │                                                SHAP vs Grad-CAM side by side
    │                                                XAI agreement summary table
    │                                                Final summary dashboard
    │
    ├── models/
    │   ├── xgb\_final\_model.pkl                   <- Trained XGBoost model
    │   ├── scaler.pkl                            <- StandardScaler (must load with model)
    │   ├── xgb\_best\_params.json                  <- Best hyperparameters from tuning
    │   └── best\_cnn\_bilstm.pth                   <- Trained CNN+BiLSTM weights
    │
    ├── artifacts/
    │   ├── X\_train\_scaled.npy                    <- Scaled training features (1152, 160)
    │   ├── X\_test\_scaled.npy                     <- Scaled test features (288, 160)
    │   ├── y\_train.npy                           <- Training labels (1152,)
    │   ├── y\_test.npy                            <- Test labels (288,)
    │   ├── y\_pred.npy                            <- XGBoost predictions (288,)
    │   └── shap\_values.npy                       <- Computed SHAP values (288, 160, 8)
    │
    └── outputs/
        ├── confusion\_matrix.png                  <- XGBoost confusion matrix
        ├── per\_emotion\_f1.png                    <- XGBoost F1 per emotion
        ├── shap\_global\_importance.png            <- SHAP global feature importance
        ├── shap\_dot\_happy.png                    <- SHAP dot plot for happy emotion
        ├── shap\_per\_emotion.png                  <- SHAP top 10 features per emotion
        ├── shap\_waterfall\_misclassified.png      <- SHAP waterfall misclassification
        ├── shap\_heatmap\_angry.png                <- SHAP heatmap for angry emotion
        ├── cnnbilstm\_confusion\_matrix.png        <- CNN+BiLSTM confusion matrix
        ├── cnn\_bilstm\_history.png                <- CNN+BiLSTM training curves
        ├── gradcam\_per\_emotion.png               <- Grad-CAM heatmaps per emotion
        ├── gradcam\_happy\_misclassified.png       <- Grad-CAM misclassification analysis
        ├── shap\_vs\_gradcam\_comparison.png        <- SHAP vs Grad-CAM comparison
        ├── final\_model\_comparison.png            <- Overall model performance comparison
        ├── final\_per\_emotion\_comparison.png      <- Per emotion model comparison
        ├── xai\_agreement\_table.png               <- XAI agreement summary table
        ├── final\_summary\_dashboard.png           <- Complete project summary dashboard
        └── final\_results.json                    <- All final metrics
```

\---

## Workflow

|Stage|Description|Notebook|
|-|-|-|
|**1. Preprocessing**|Extract labels, standardize audio, extract MFCC (160 features) and Mel Spectrograms|Preprocessing.ipynb + 01\_preprocessing\_v2.ipynb|
|**2. Classical ML**|Train XGBoost on 160 MFCC features, hyperparameter tuning, evaluation|02\_classical\_ml\_shap.ipynb|
|**3. SHAP Analysis**|Global importance, per-emotion plots, waterfall, heatmap|02\_classical\_ml\_shap.ipynb|
|**4. Deep Learning**|Train CNN + BiLSTM on MFCC sequences, evaluation|03\_cnn\_gradcam.ipynb|
|**5. Grad-CAM Analysis**|Per-emotion heatmaps, misclassification analysis|03\_cnn\_gradcam.ipynb|
|**6. Comparison**|SHAP vs Grad-CAM, model performance comparison|04\_cnn\_bilstm\_gradcam\_comparison\_conclusion.ipynb|
|**7. Conclusion**|Key findings, limitations, future work|04\_cnn\_bilstm\_gradcam\_comparison\_conclusion.ipynb|

### Summary

* **MFCC features (160)** are used for both XGBoost and CNN + BiLSTM
* **Mel Spectrograms** were extracted but not used in final models — 2D CNN was attempted but abandoned due to insufficient data (1440 samples)
* **XGBoost F1 = 0.630** explained by SHAP
* **CNN + BiLSTM F1 = 0.614** explained by Grad-CAM
* **SHAP and Grad-CAM agree on 5 out of 6 key findings (83%)**

\---

## How to Reproduce Results

### Step 1 — Download Dataset

Download the RAVDESS speech audio files from:
https://zenodo.org/records/1188976

Extract to:

```
Project/data/raw/
    Actor\_01/
    Actor\_02/
    ...
    Actor\_24/
```

### Step 2 — Run Preprocessing

```
Run: XAI\_2026/notebooks/Preprocessing.ipynb
Run: XAI\_2026/notebooks/01\_preprocessing\_v2.ipynb
```

This generates all files in `data/processed/`

### Step 3 — Run Phase 1 (XGBoost + SHAP)

```
Run: XAI\_2026/notebooks/02\_classical\_ml\_shap.ipynb
```

Saves models to `XAI\_2026/models/`
Saves plots to `XAI\_2026/outputs/`

### Step 4 — Run Phase 2 (CNN+BiLSTM + Grad-CAM)

```
Run: XAI\_2026/notebooks/03\_cnn\_gradcam.ipynb
```

Saves model to `XAI\_2026/models/`
Saves plots to `XAI\_2026/outputs/`

### Step 5 — Run Phase 3 (Comparison + Conclusion)

```
Run: XAI\_2026/notebooks/04\_cnn\_bilstm\_gradcam\_comparison\_conclusion.ipynb
```

Saves all comparison plots to `XAI\_2026/outputs/`

\---

## Installation

```bash
pip install numpy==1.26.4 pandas==2.2.3 tensorflow==2.17.0
pip install torch scikit-learn xgboost shap librosa joblib matplotlib seaborn jupyter
```

\---

## Results Summary

### Model Performance

|Model|Accuracy|Weighted F1|XAI Method|
|-|-|-|-|
|XGBoost|63.2%|0.630|SHAP|
|CNN + BiLSTM|62.0%|0.614|Grad-CAM|

### Per Emotion F1

|Emotion|XGBoost|CNN+BiLSTM|Better Model|
|-|-|-|-|
|Angry|0.64|0.58|XGBoost|
|Calm|0.61|0.69|CNN+BiLSTM|
|Disgust|0.67|0.65|XGBoost|
|Fear|0.70|0.72|CNN+BiLSTM|
|Happy|0.54|0.47|XGBoost|
|Neutral|0.56|0.47|XGBoost|
|Sad|0.61|0.52|XGBoost|
|Surprise|0.69|0.72|CNN+BiLSTM|

### XAI Agreement

|Finding|SHAP|Grad-CAM|Agreement|
|-|-|-|-|
|Low MFCCs most important|MFCC 1,2,3 dominate|MFCC 0-2 activated|AGREE|
|Angry signature|delta\_std\_1 defines angry|MFCC 5-9 concentrated|AGREE|
|Happy/Fear confusion|MFCC 8 pushes from happy|MFCC 8 triggered fear|STRONGLY AGREE|
|Emotion profiles|Each emotion unique pattern|Each emotion unique range|AGREE|
|Higher MFCC usage|Drops after index 10|Active through index 32|DISAGREE|
|Happy/Neutral difficulty|Scattered weak patterns|No concentrated signal|AGREE|

**Overall agreement rate: 5 out of 6 findings (83%)**

\---

## Key Findings

1. Low frequency MFCC bands (1 through 3) carry the most emotional information, confirmed independently by both SHAP and Grad-CAM.
2. Temporal dynamics are equally important as static tonal features. mfcc\_delta\_std\_1 is the second most globally important feature, nearly equal to mfcc\_mean\_3.
3. MFCC 8 is the critical feature driving happy-to-fear misclassification. Both XAI methods independently identified this same feature — the strongest finding in the project.
4. Happy is acoustically ambiguous. Neither model nor XAI method finds a clear separating feature profile for happy, explaining its consistently low F1 across both models.
5. Classical ML and deep learning achieve similar accuracy on this small dataset. XGBoost outperforms CNN+BiLSTM on emotions defined by static features. CNN+BiLSTM outperforms on emotions with strong temporal patterns.

\---

## Limitations

* Dataset size of 1440 samples limits deep learning potential
* 2D CNN on mel spectrograms was attempted but abandoned due to insufficient data
* No leave-one-actor-out evaluation so speaker identity may influence results
* Mean MFCC features lose some temporal detail despite delta augmentation
* Neutral class is underrepresented with only 96 samples versus 192 for other emotions

\---

## Notes on File Availability

The raw `.wav` files are not tracked in Git. The mel spectrogram file `ravdess\_mel\_spectrograms.npy` (\~207 MB) is available on the shared project drive and is not tracked in Git due to file size. All other processed files in `data/processed/` can be regenerated by running the preprocessing notebooks on the raw dataset.

https://drive.google.com/drive/folders/1lU-FfvX7Fv0ldasEfXXTSrSOHLNDsZR0?usp=drive_link

