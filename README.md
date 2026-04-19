#  Titanic Passenger Survival — Complete Data Analysis Project

> A comprehensive end-to-end data analysis pipeline: data loading, cleaning,
> transformation, exploratory data analysis (EDA), visualization, and documentation.

---

##  Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Setup & Installation](#setup--installation)
5. [How to Run](#how-to-run)
6. [Steps Covered](#steps-covered)
7. [Visualizations](#visualizations)
8. [Key Findings](#key-findings)
9. [Git Workflow](#git-workflow)

---

## Project Overview

This project applies a full data analysis workflow to the Titanic passenger dataset.
The goal is to uncover factors that influenced survival and demonstrate professional
data analysis techniques including cleaning, transformation, EDA, and visualization.

---

## Dataset

| Property       | Detail                                      |
|----------------|---------------------------------------------|
| Name           | Titanic Passenger Records                   |
| Rows           | 891 passengers                              |
| Raw Columns    | 12                                          |
| Final Columns  | 25 (after cleaning + feature engineering)   |
| Target         | `survived` (0 = Died, 1 = Survived)         |
| Missing Data   | `age` (19.4%), `deck` (76.5%)               |

**Raw Features:** name, survived, pclass, sex, age, sibsp, parch, fare, embarked,
embark_town, who, deck

---

## Project Structure

```
titanic-data-analysis/
│
├── README.md                          ← This file
├── analysis.py                        ← Main Python script (all steps)
├── GIT_COMMANDS.txt                   ← Step-by-step GitHub guide
├── .gitignore
│
├── data/
│   ├── titanic_raw.csv                ← Original unmodified dataset
│   └── titanic_cleaned.csv            ← Cleaned & feature-engineered dataset
│
├── outputs/
│   ├── fig1_distributions.png         ← Histograms + KDE density plots
│   ├── fig2_survival_dashboard.png    ← Survival rates across all groups
│   ├── fig3_relationships.png         ← Scatter plots & grouped analysis
│   ├── fig4_pair_plot.png             ← Multi-variable pair plot
│   ├── fig5_correlation.png           ← Heatmap + ranked correlation bar
│   ├── fig6_outlier_boxplots.png      ← Before/after outlier treatment
│   ├── fig7_feature_engineering.png   ← New feature analysis & impact
│   ├── key_insights.txt               ← 8 key findings summary
│   └── process_documentation.txt      ← Full methodology report
│
└── notebooks/
    └── titanic_analysis.ipynb         ← Jupyter Notebook (interactive)
```

---

## Setup & Installation

### Requirements
- Python 3.8+
- pip

### Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Or install from requirements:
```bash
pip install -r requirements.txt
```

**requirements.txt contents:**
```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.1.0
jupyter>=1.0.0
```

---

## How to Run

### Option A — Python Script
```bash
python analysis.py
```

### Option B — Jupyter Notebook
```bash
jupyter notebook notebooks/titanic_analysis.ipynb
```

All outputs are automatically saved to the `outputs/` directory.

---

## Steps Covered

###  Step 1 — Data Collection & Exploration
- Loaded dataset into Pandas DataFrame
- Examined shape, dtypes, column info (non-null counts, unique values)
- Viewed first 5 and last 5 rows
- Generated full numerical summary statistics (`describe()`)
- Value counts for all categorical columns
- Identified missing values: `age` (19.4%), `deck` (76.5%)
- Detected 0 duplicate rows
- IQR-based outlier detection for: age, fare, sibsp, parch

###  Step 2 — Data Cleaning & Transformation

**Missing Values:**
| Column   | Method                                | Reason                             |
|----------|---------------------------------------|------------------------------------|
| `age`    | Grouped median (pclass × sex)         | Respects subgroup distributions    |
| `embarked` | Mode imputation                     | Only 0–2 values missing            |
| `deck`   | Column dropped                        | 76.5% missing — unrecoverable      |

**Outlier Treatment:**
| Column | Method                          | Threshold  |
|--------|---------------------------------|------------|
| `fare` | 99th percentile capping         | £480.45    |
| others | No treatment (genuine values)   | —          |

**Encoding:**
| Column     | Method        | Result                           |
|------------|---------------|----------------------------------|
| `sex`      | Label Encode  | female=0, male=1                 |
| `embarked` | Label Encode  | C=0, Q=1, S=2                    |
| `pclass`   | One-Hot       | pclass_1, pclass_2, pclass_3     |

**Scaling:** MinMax normalization [0, 1] applied to: age, fare, sibsp, parch

**Feature Engineering (6 new features):**
| Feature           | Formula / Logic              | Purpose                        |
|-------------------|------------------------------|--------------------------------|
| `family_size`     | sibsp + parch + 1            | Total group size               |
| `is_alone`        | 1 if family_size == 1        | Solo traveller indicator       |
| `age_group`       | Bins: 0–12/12–18/18–35/35–60/60+ | Non-linear age effect     |
| `title`           | Mapped from `who` column     | Social status proxy            |
| `fare_per_person` | fare / family_size           | True cost per individual       |
| `fare_band`       | Quartile-based buckets       | Categorical wealth tier        |

###  Step 3 — Exploratory Data Analysis

**Descriptive Statistics:**
- Full numerical stats for 6 key variables
- Frequency tables + percentage for all categorical variables
- Survival rates computed for every categorical dimension

**7 Visualization Figures:**

| Figure | Type | Description |
|--------|------|-------------|
| Fig 1  | Histograms + KDE | Distributions of all key variables |
| Fig 2  | Bar + Pie charts | Survival dashboard across all groups |
| Fig 3  | Scatter + Bar + KDE | Variable relationships |
| Fig 4  | Pair Plot | Multi-variable pairwise relationships |
| Fig 5  | Heatmap + Bar | Full correlation matrix |
| Fig 6  | Box Plots | Outlier detection before & after |
| Fig 7  | Mixed | Feature engineering impact analysis |

###  Step 7 — Documentation
- Inline comments throughout `analysis.py` explaining every step
- `key_insights.txt` — 8 structured findings
- `process_documentation.txt` — full methodology report
- This README file

###  Step 8 — Version Control
- `GIT_COMMANDS.txt` — step-by-step git init → GitHub push guide
- Recommended commit message conventions included
- `.gitignore` setup covered

---

## Visualizations

| Figure | Preview Description |
|--------|---------------------|
| Fig 1 — Distributions | Age & fare histograms overlaid with KDE curves; family size, sibsp, parch counts; pie chart for embarkation ports |
| Fig 2 — Survival Dashboard | 9-panel dashboard showing survival by sex, class, age group, embarkation, alone/family, fare band, title, family size |
| Fig 3 — Relationships | Scatter (age vs fare coloured by survival/class), grouped bars (sex×class, sex×embarkation), KDE density overlay |
| Fig 4 — Pair Plot | Pairwise scatter + KDE diagonal for survived/age/fare/family_size/fare_per_person |
| Fig 5 — Correlation | Lower-triangle heatmap of 13 features; ranked horizontal bar chart vs survived |
| Fig 6 — Box Plots | 2-row grid: raw vs cleaned box plots for fare, age, sibsp, parch with quartile annotations |
| Fig 7 — Features | Fare band scatter, fare/person by class violin, age×sex stacked bar, is_alone impact, title dual-axis, scaled density |

---

## Key Findings

1. **Gender dominates** — Women survived at 91.7% vs men at 5.0% (86.8 pt gap)
2. **Class inequality** — 1st class (48.9%) vs 3rd class (33.6%) survival
3. **Children prioritised** — Under-12s survived at 58.7%
4. **Fare proxies wealth** — Premium band: 45.7% vs Budget: 30.5% survival
5. **Cherbourg advantage** — 44.9% vs Southampton 34.4% (class composition effect)
6. **Family effect** — Small families (2–4) survived best; very large families worst
7. **Data quality** — 19.4% age imputed; deck column dropped (76.5% missing)
8. **Best ML features** — sex, pclass, age, fare, family_size, fare_per_person

---

## Git Workflow

```bash
# Clone the repository
git clone https://github.com/<your-username>/titanic-data-analysis.git
cd titanic-data-analysis

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the analysis
python analysis.py
```

See `GIT_COMMANDS.txt` for the full push workflow.

---

