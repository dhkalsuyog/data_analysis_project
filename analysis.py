# SECTION 1 — IMPORT LIBRARIES


import warnings
warnings.filterwarnings('ignore')       # Suppress non-critical warnings

import pandas as pd                     # DataFrames and data manipulation
import numpy as np                      # Arrays and numerical operations
import matplotlib                       # Base plotting library
matplotlib.use('Agg')                   # Non-interactive backend (saves PNGs)
import matplotlib.pyplot as plt         # pyplot interface for creating plots
import matplotlib.gridspec as gridspec  # Custom multi-panel subplot layouts
import matplotlib.patches as mpatches   # Custom legend markers
import seaborn as sns                   # Statistical visualization library

from sklearn.preprocessing import (
    LabelEncoder,   # Converts text categories to integer labels (e.g. male=1)
    MinMaxScaler    # Scales numerical values to the range [0, 1]
)


# SECTION 2 — GLOBAL SETTINGS & CONFIGURATION
# Centralizing all settings here means we only need to change one place
# if we want to update colours, paths, or styles across the whole project.

# ── File Paths 
BASE_DIR   = 'data_analysis_project'    # Root project folder
DATA_DIR   = f'{BASE_DIR}/data'         # Raw and cleaned datasets
OUTPUT_DIR = f'{BASE_DIR}/outputs'      # All figures and report files

# ── Colour Palette 
# A consistent 6-colour palette used in every figure for visual consistency.
PALETTE = [
    '#2E86AB',   # Blue   — primary colour
    '#E84855',   # Red    — highlights / "died" category
    '#3BB273',   # Green  — positive / "survived" category
    '#F4A261',   # Orange — secondary accent
    '#9B5DE5',   # Purple — tertiary accent
    '#F15BB5',   # Pink   — additional category
]

#  Visual Theme 
BG_COLOR = '#F8F9FB'    # Soft off-white background — easier on the eyes
sns.set_theme(style='whitegrid', palette=PALETTE)
plt.rcParams.update({
    'figure.facecolor' : BG_COLOR,
    'axes.facecolor'   : BG_COLOR,
    'axes.grid'        : True,
    'grid.color'       : '#E2E8F0',
    'grid.linewidth'   : 0.6,
    'font.family'      : 'DejaVu Sans',
    'axes.titlesize'   : 13,
    'axes.titleweight' : 'bold',
    'axes.titlepad'    : 10,
    'axes.labelsize'   : 11,
    'xtick.labelsize'  : 9,
    'ytick.labelsize'  : 9,
    'legend.fontsize'  : 9,
})


#  Helper Functions 

def print_banner(title, width=70):
    """
    Prints a formatted section banner to the console.
    Makes the terminal output easy to read and navigate.
    """
    print('\n' + '═' * width)
    print(f'  {title}')
    print('═' * width)


def save_figure(fig, filename, dpi=150):
    """
    Saves a matplotlib figure to outputs/ directory as a PNG file.
    Closes the figure after saving to free memory.

    Parameters:
        fig      : matplotlib Figure object to save
        filename : output filename  e.g. 'fig1_distributions.png'
        dpi      : image resolution (150 = good quality)
    """
    filepath = f'{OUTPUT_DIR}/{filename}'
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)      # Free memory after saving
    print(f'    Saved → outputs/{filename}')



# SECTION 3 — STEP 1: DATA LOADING & EXPLORATION
# GOAL: Understand the raw dataset BEFORE making any changes.
#       Examine structure, distributions, and find data quality problems.


print_banner('STEP 1 — DATA LOADING & EXPLORATION')

#  1A. Load Dataset 
# Load the CSV into a pandas DataFrame.
# A DataFrame is a table — rows = passengers, columns = features.
# We keep df_raw untouched and work on df (a copy).
RAW_PATH = f'{DATA_DIR}/titanic_raw.csv'
df_raw   = pd.read_csv(RAW_PATH)    # Load raw data
df       = df_raw.copy()            # Working copy — never modify df_raw

print(f'\n  File loaded : {RAW_PATH}')
print(f'  Shape       : {df.shape[0]:,} rows  ×  {df.shape[1]} columns')

#  1B. Examine Dataset Structure 
# We look at column names, data types, and how many values are missing.
print('\n  COLUMN STRUCTURE ───────────────')
column_info = pd.DataFrame({
    'Data Type'    : df.dtypes,
    'Non-Null'     : df.notnull().sum(),    # Count of filled values
    'Missing'      : df.isnull().sum(),     # Count of empty cells
    'Unique Values': df.nunique(),          # How many distinct values exist
})
print(column_info.to_string())

# Show first 5 rows — gives a feel for what the data looks like
print('\n  FIRST 5 ROWS ───────────────')
print(df.head(5).to_string())

# Show last 5 rows — checks for anomalies at the end of the file
print('\n  LAST 5 ROWS ──────────────────')
print(df.tail(5).to_string())

# ── 1C. Summary Statistics ────────────────────────────────────────────────────
# describe() gives count, mean, std, min, quartiles, max for numeric columns.
# This reveals the range and spread of each numerical variable.
print('\n  NUMERICAL SUMMARY STATISTICS ─────────────────')
print(df.describe(include=[np.number]).round(2).to_string())

# For text columns, count how many of each value exist
print('\n  ── CATEGORICAL VALUE COUNTS ──────────────')
for col in df.select_dtypes(include='object').columns:
    print(f'\n  [ {col} ]')
    print(df[col].value_counts(dropna=False).to_string())

# ── 1D. Identify Data Problems ───────────────────────────────────────────────
# Three things to check: missing values, duplicates, outliers

# Problem 1: Missing Values
print('\n   MISSING VALUES REPORT ────────────────')
missing_count   = df.isnull().sum()
missing_percent = (missing_count / len(df) * 100).round(2)
missing_report  = pd.DataFrame({
    'Missing Count': missing_count,
    'Missing %'    : missing_percent
})
# Only show columns that have missing values
problem_cols = missing_report[missing_report['Missing Count'] > 0]
print(problem_cols.sort_values('Missing %', ascending=False).to_string())

# Problem 2: Duplicate Rows
print('\n   DUPLICATE ROWS ────────')
print(f'  Duplicate rows found: {df.duplicated().sum()}')

# Problem 3: Outliers — IQR (Interquartile Range) method
# Values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR are flagged as outliers
print('\n  ── OUTLIER DETECTION (IQR Method) ───────────')
print(f'  {"Column":<12} {"Lower Bound":>13} {"Upper Bound":>13} {"Outliers":>10}')
print('  ' + '-'*52)
for col in ['age', 'fare', 'sibsp', 'parch']:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo  = Q1 - 1.5 * IQR
    hi  = Q3 + 1.5 * IQR
    n   = ((df[col] < lo) | (df[col] > hi)).sum()
    print(f'  {col:<12} {lo:>13.2f} {hi:>13.2f} {n:>10}')



# SECTION 4 — STEP 2: DATA CLEANING & TRANSFORMATION
# GOAL: Fix all data quality problems. Then transform and enrich the data.


print_banner('STEP 2 — DATA CLEANING & TRANSFORMATION')

#2A. Handle Missing Values 

print('\n  [2A] HANDLING MISSING VALUES')
print('  ' + '-'*55)

# Fill missing AGE with median of its pclass × sex group
df['age'] = df.groupby(['pclass', 'sex'])['age'].transform(
    lambda x: x.fillna(x.median())
)
print('   age      → Grouped median imputation by (pclass × sex)')

# Fill missing EMBARKED with the most common port
if df['embarked'].isnull().sum() > 0:
    mode_embarked = df['embarked'].mode()[0]
    df['embarked']    = df['embarked'].fillna(mode_embarked)
    df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])
    print(f'   embarked → Mode imputation  (filled with "{mode_embarked}")')

# Drop DECK column — too many missing values
if 'deck' in df.columns:
    df.drop(columns=['deck'], inplace=True)
    print('   deck     → Column dropped (76.5% missing)')

print(f'\n  Missing values after cleaning: {df.isnull().sum().sum()} ✅')

# ── 2B. Outlier Treatment ─────────────────────────────────────────────────────

print('\n  [2B] OUTLIER TREATMENT')
print('  ' + '-'*55)

df['fare_raw']       = df['fare'].copy()     # Save original for comparison
fare_99th_percentile = df['fare'].quantile(0.99)
capped_count         = (df['fare'] > fare_99th_percentile).sum()
df['fare']           = df['fare'].clip(upper=fare_99th_percentile)

print(f'   fare   → Capped at 99th percentile (£{fare_99th_percentile:.2f})')
print(f'             {capped_count} extreme values capped')
print('   age    → No treatment (0.5–80 yrs is valid)')
print('   sibsp  → No treatment (large values = genuine big families)')
print('   parch  → No treatment (large values = genuine big families)')

#2C. Categorical Encoding 

print('\n  [2C] CATEGORICAL ENCODING')
print('  ' + '-'*55)

le = LabelEncoder()

df['sex_encoded']      = le.fit_transform(df['sex'])       # female=0, male=1
print('   sex      → Label encoded  (female=0, male=1)')

df['embarked_encoded'] = le.fit_transform(df['embarked'])  # C=0, Q=1, S=2
print('   embarked → Label encoded  (C=0, Q=1, S=2)')

df = pd.get_dummies(df, columns=['pclass'], prefix='pclass', drop_first=False)
for col in ['pclass_1', 'pclass_2', 'pclass_3']:
    if col in df.columns:
        df[col] = df[col].astype(int)   # Convert True/False to 1/0
print('   pclass   → One-hot encoded (pclass_1, pclass_2, pclass_3)')

# 2D. Numerical Scaling 


print('\n  [2D] NUMERICAL SCALING (MinMax → [0.0 to 1.0])')
print('  ' + '-'*55)

scaler          = MinMaxScaler()
cols_to_scale   = ['age', 'fare', 'sibsp', 'parch']
scaled_array    = scaler.fit_transform(df[cols_to_scale])

for i, col in enumerate(cols_to_scale):
    new_col        = f'{col}_scaled'
    df[new_col]    = scaled_array[:, i]
    print(f'   {col:<8} → {new_col}')


print('\n  [2E] FEATURE ENGINEERING — 6 New Columns Created')
print('  ' + '-'*55)

# 1. Total family size (siblings + parents/children + self)
df['family_size'] = df['sibsp'] + df['parch'] + 1
print('   family_size    = sibsp + parch + 1')

# 2. Is the passenger alone on the ship?
df['is_alone'] = (df['family_size'] == 1).astype(int)
print('   is_alone       = 1 if family_size==1, else 0')

# 3. Age group — non-linear age bins capture life-stage thresholds
df['age_group'] = pd.cut(
    df['age'],
    bins   = [0, 12, 18, 35, 60, 100],
    labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
)
print('   age_group      = Child / Teen / Young Adult / Adult / Senior')

# 4. Title — derived from the "who" column as a social status proxy
df['title'] = df['who'].map({
    'man'   : 'Mr',
    'woman' : 'Mrs/Miss',
    'child' : 'Master'
})
print('   title          = Mr / Mrs/Miss / Master')

# 5. True cost per person — fare divided by total family members
df['fare_per_person'] = df['fare'] / df['family_size']
print('   fare_per_person = fare / family_size')

# 6. Wealth tier — quartile-based fare categories
df['fare_band'] = pd.qcut(
    df['fare'],
    q      = 4,
    labels = ['Budget', 'Economy', 'Business', 'Premium']
)
print('  fare_band      = Budget / Economy / Business / Premium')

print(f'\n  Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns')

# Save cleaned dataset
CLEAN_PATH = f'{DATA_DIR}/titanic_cleaned.csv'
df.to_csv(CLEAN_PATH, index=False)
print(f'   Cleaned dataset saved → data/titanic_cleaned.csv')



# SECTION 5 — STEP 3A: DESCRIPTIVE STATISTICS (EDA)
# GOAL: Summarise the cleaned data with statistics and survival breakdowns.

print_banner('STEP 3A — DESCRIPTIVE STATISTICS')

# Numerical summary for key features
print('\n  Numerical Variables — Summary Statistics:')
num_feats = ['age', 'fare', 'sibsp', 'parch', 'family_size', 'fare_per_person']
print(df[num_feats].describe().round(2).to_string())

# Frequency tables for categorical columns
print('\n  Categorical Variables — Frequency Tables:')
for col in ['sex', 'embarked', 'age_group', 'title', 'fare_band', 'is_alone']:
    counts = df[col].value_counts(dropna=False)
    pct    = (counts / len(df) * 100).round(1)
    print(f'\n  [ {col} ]')
    print(pd.DataFrame({'Count': counts, 'Pct %': pct}).to_string())

# Survival rate broken down by every key group
print('\n  Survival Rates by Group:')
for col in ['sex', 'who', 'age_group', 'title', 'fare_band', 'is_alone', 'embarked']:
    sr = df.groupby(col, observed=True)['survived'].agg(['mean', 'count'])
    sr.columns = ['Survival Rate', 'Count']
    sr['Survival Rate'] = (sr['Survival Rate'] * 100).round(1).astype(str) + '%'
    print(f'\n  [ {col} ]')
    print(sr.to_string())



# SECTION 6 — STEP 3B: VISUALIZATIONS
# GOAL: Create 7 professional figures that visually reveal patterns in the data.


print_banner('STEP 3B — GENERATING 7 VISUALIZATION FIGURES')



# FIGURE 1 — DATA DISTRIBUTIONS (Histograms + KDE Density Plots)

print('\n  Generating Figure 1: Distributions …')

fig = plt.figure(figsize=(20, 14), facecolor=BG_COLOR)
fig.suptitle('Figure 1 — Data Distributions: Histograms & KDE Density Plots',
             fontsize=16, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

# Chart 1-1: Age — Histogram + KDE overlay
# Histogram shows count per age bucket. KDE shows smooth overall shape.
ax = fig.add_subplot(gs[0, 0])
df['age'].hist(bins=35, ax=ax, color=PALETTE[0], edgecolor='white', alpha=0.80)
ax2 = ax.twinx()                    # Second y-axis for the KDE density curve
df['age'].plot.kde(ax=ax2, color=PALETTE[1], linewidth=2.5)
ax.axvline(df['age'].mean(),   color='red',   linestyle='--', lw=2,
           label=f'Mean {df["age"].mean():.1f} yrs')
ax.axvline(df['age'].median(), color='green', linestyle='--', lw=2,
           label=f'Median {df["age"].median():.1f} yrs')
ax.set_title('Age — Histogram + KDE')
ax.set_xlabel('Age (years)')
ax.set_ylabel('Passenger Count')
ax2.set_ylabel('Density', color=PALETTE[1])
ax.legend(fontsize=8)

# Chart 1-2: Fare — Histogram + KDE overlay
ax = fig.add_subplot(gs[0, 1])
df['fare'].hist(bins=35, ax=ax, color=PALETTE[3], edgecolor='white', alpha=0.80)
ax3 = ax.twinx()
df['fare'].plot.kde(ax=ax3, color=PALETTE[1], linewidth=2.5)
ax.axvline(df['fare'].mean(), color='red', linestyle='--', lw=2,
           label=f'Mean £{df["fare"].mean():.1f}')
ax.set_title('Fare (£) — Histogram + KDE')
ax.set_xlabel('Fare (£)')
ax.set_ylabel('Passenger Count')
ax3.set_ylabel('Density', color=PALETTE[1])
ax.legend(fontsize=8)

# Chart 1-3: Family Size — Bar Chart
ax = fig.add_subplot(gs[0, 2])
fs = df['family_size'].value_counts().sort_index()
ax.bar(fs.index, fs.values, color=PALETTE[4], edgecolor='white', width=0.7)
for x, y in zip(fs.index, fs.values):
    ax.text(x, y + 3, str(y), ha='center', fontsize=8, fontweight='bold')
ax.set_title('Family Size — Distribution')
ax.set_xlabel('Family Size')
ax.set_ylabel('Passenger Count')

# Chart 1-4: Siblings/Spouses Count
ax = fig.add_subplot(gs[1, 0])
sb = df['sibsp'].value_counts().sort_index()
ax.bar(sb.index, sb.values, color=PALETTE[0], edgecolor='white', width=0.7)
ax.set_title('Siblings / Spouses Aboard (sibsp)')
ax.set_xlabel('Count')
ax.set_ylabel('Passenger Count')

# Chart 1-5: Parents/Children Count
ax = fig.add_subplot(gs[1, 1])
pc = df['parch'].value_counts().sort_index()
ax.bar(pc.index, pc.values, color=PALETTE[2], edgecolor='white', width=0.7)
ax.set_title('Parents / Children Aboard (parch)')
ax.set_xlabel('Count')
ax.set_ylabel('Passenger Count')

# Chart 1-6: Fare Per Person — Pure KDE Density Plot
# No histogram bars — just the smooth density curve for this engineered feature
ax = fig.add_subplot(gs[1, 2])
df['fare_per_person'].plot.kde(ax=ax, color=PALETTE[5], linewidth=2.5)
ax.fill_between(ax.lines[0].get_xdata(), ax.lines[0].get_ydata(),
                alpha=0.15, color=PALETTE[5])    # Shaded area under curve
ax.set_title('Fare Per Person — KDE Density')
ax.set_xlabel('Fare Per Person (£)')
ax.set_ylabel('Density')
ax.set_xlim(0, df['fare_per_person'].quantile(0.98))

# Chart 1-7: Sex — Bar Chart
ax = fig.add_subplot(gs[2, 0])
sx = df['sex'].value_counts()
bars = ax.bar(sx.index.str.capitalize(), sx.values,
              color=[PALETTE[4], PALETTE[0]], edgecolor='white', width=0.5)
for bar, v in zip(bars, sx.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+4,
            f'{v}\n({v/len(df)*100:.1f}%)', ha='center', fontsize=9, fontweight='bold')
ax.set_title('Sex — Distribution')
ax.set_ylabel('Passenger Count')
ax.set_ylim(0, sx.max() * 1.25)

# Chart 1-8: Embarkation Port — Pie Chart
ax = fig.add_subplot(gs[2, 1])
em = df['embark_town'].value_counts()
_, _, autotexts = ax.pie(em.values, labels=em.index, autopct='%1.1f%%',
                          colors=PALETTE[:3], startangle=90,
                          wedgeprops={'edgecolor': 'white', 'linewidth': 2})
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight('bold')
ax.set_title('Embarkation Port — Proportions')

# Chart 1-9: Passenger Class — Bar Chart
ax = fig.add_subplot(gs[2, 2])
cl = df_raw['pclass'].value_counts().sort_index()
bars = ax.bar(['1st Class', '2nd Class', '3rd Class'], cl.values,
              color=PALETTE[:3], edgecolor='white', width=0.6)
for bar, v in zip(bars, cl.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
            str(v), ha='center', fontsize=11, fontweight='bold')
ax.set_title('Passenger Class — Distribution')
ax.set_ylabel('Passenger Count')

save_figure(fig, 'fig1_distributions.png')


# FIGURE 2 — SURVIVAL OVERVIEW DASHBOARD (9 Charts)


print('  Generating Figure 2: Survival Dashboard …')

fig = plt.figure(figsize=(20, 14), facecolor=BG_COLOR)
fig.suptitle('Figure 2 — Survival Overview Dashboard',
             fontsize=16, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

# Chart 2-1: Overall Survival Count
ax = fig.add_subplot(gs[0, 0])
sv = df['survived'].value_counts().rename({0: 'Died', 1: 'Survived'})
bars = ax.bar(sv.index, sv.values,
              color=[PALETTE[1], PALETTE[2]], edgecolor='white', width=0.5)
for bar, v in zip(bars, sv.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
            f'{v}\n({v/len(df)*100:.1f}%)', ha='center', fontsize=11, fontweight='bold')
ax.set_title('Overall Survival — Died vs Survived')
ax.set_ylabel('Number of Passengers')
ax.set_ylim(0, sv.max() * 1.25)

# Chart 2-2: Survival Rate by Sex
ax = fig.add_subplot(gs[0, 1])
sex_sv = df.groupby('sex')['survived'].mean() * 100
bars = ax.bar(sex_sv.index.str.capitalize(), sex_sv.values,
              color=[PALETTE[4], PALETTE[0]], edgecolor='white', width=0.5)
for bar, v in zip(bars, sex_sv.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f'{v:.1f}%', ha='center', fontsize=13, fontweight='bold')
ax.set_title('Survival Rate by Sex')
ax.set_ylabel('Survival Rate (%)')
ax.set_ylim(0, 110)

# Chart 2-3: Survival Rate by Passenger Class
ax = fig.add_subplot(gs[0, 2])
cls_sv = df_raw.groupby('pclass')['survived'].mean() * 100
bars = ax.bar(['1st Class', '2nd Class', '3rd Class'], cls_sv.values,
              color=PALETTE[:3], edgecolor='white', width=0.6)
for bar, v in zip(bars, cls_sv.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f'{v:.1f}%', ha='center', fontsize=13, fontweight='bold')
ax.set_title('Survival Rate by Passenger Class')
ax.set_ylabel('Survival Rate (%)')
ax.set_ylim(0, 110)

# Chart 2-4: Survival Rate by Age Group
ax = fig.add_subplot(gs[1, 0])
ag_sv = df.groupby('age_group', observed=True)['survived'].mean() * 100
bars = ax.bar(ag_sv.index, ag_sv.values,
              color=PALETTE[:len(ag_sv)], edgecolor='white')
for bar, v in zip(bars, ag_sv.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{v:.0f}%', ha='center', fontsize=9, fontweight='bold')
ax.set_title('Survival Rate by Age Group')
ax.set_xlabel('Age Group')
ax.set_ylabel('Survival Rate (%)')
ax.tick_params(axis='x', rotation=15)
ax.set_ylim(0, 110)

# Chart 2-5: Survival Rate by Embarkation Port
ax = fig.add_subplot(gs[1, 1])
pt_sv = df.groupby('embark_town')['survived'].mean() * 100
bars = ax.bar(pt_sv.index, pt_sv.values,
              color=PALETTE[:3], edgecolor='white')
for bar, v in zip(bars, pt_sv.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_title('Survival Rate by Embarkation Port')
ax.set_ylabel('Survival Rate (%)')
ax.set_ylim(0, 110)

# Chart 2-6: Survival — Alone vs With Family
ax = fig.add_subplot(gs[1, 2])
al_sv = df.groupby('is_alone')['survived'].mean() * 100
bars = ax.bar(['With Family', 'Alone'], al_sv.values,
              color=[PALETTE[2], PALETTE[1]], edgecolor='white', width=0.5)
for bar, v in zip(bars, al_sv.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f'{v:.1f}%', ha='center', fontsize=13, fontweight='bold')
ax.set_title('Survival: Alone vs With Family')
ax.set_ylabel('Survival Rate (%)')
ax.set_ylim(0, 110)

# Chart 2-7: Survival by Fare Band
ax = fig.add_subplot(gs[2, 0])
fb_sv = df.groupby('fare_band', observed=True)['survived'].mean() * 100
bars = ax.bar(fb_sv.index, fb_sv.values,
              color=PALETTE[:4], edgecolor='white')
for bar, v in zip(bars, fb_sv.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{v:.0f}%', ha='center', fontsize=10, fontweight='bold')
ax.set_title('Survival Rate by Fare Band (Wealth Tier)')
ax.set_xlabel('Fare Band')
ax.set_ylabel('Survival Rate (%)')
ax.set_ylim(0, 110)

# Chart 2-8: Survival by Title
ax = fig.add_subplot(gs[2, 1])
ti_sv = df.groupby('title')['survived'].mean() * 100
bars = ax.bar(ti_sv.index, ti_sv.values,
              color=PALETTE[3:6], edgecolor='white')
for bar, v in zip(bars, ti_sv.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_title('Survival Rate by Title (Social Status)')
ax.set_ylabel('Survival Rate (%)')
ax.set_ylim(0, 110)

# Chart 2-9: Survival Rate by Family Size — Line Plot
# Line plot shows the TREND as family size increases — better than a bar chart here
ax = fig.add_subplot(gs[2, 2])
fam_sv = df.groupby('family_size')['survived'].mean() * 100
ax.plot(fam_sv.index, fam_sv.values, 'o-',
        color=PALETTE[0], linewidth=2.5, markersize=9,
        markeredgecolor='white', markeredgewidth=1.5)
ax.fill_between(fam_sv.index, fam_sv.values, alpha=0.12, color=PALETTE[0])
for x, y in zip(fam_sv.index, fam_sv.values):
    ax.annotate(f'{y:.0f}%', (x, y),
                textcoords='offset points', xytext=(0, 9), ha='center', fontsize=8)
ax.set_title('Survival Rate by Family Size')
ax.set_xlabel('Family Size')
ax.set_ylabel('Survival Rate (%)')
ax.set_ylim(0, 110)

save_figure(fig, 'fig2_survival_dashboard.png')



# FIGURE 3 — RELATIONSHIP PLOTS (Scatter, Violin, Grouped Bar, KDE)

print('  Generating Figure 3: Relationship Plots …')

fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor=BG_COLOR)
fig.suptitle('Figure 3 — Relationship Plots: Scatter, Violin & Grouped Comparisons',
             fontsize=16, fontweight='bold', y=1.02)

# Chart 3-1: Scatter — Age vs Fare, coloured by Survival
# Each dot = one passenger. Green=survived, Red=died.
sc = axes[0,0].scatter(df['age'], df['fare'],
    c=df['survived'], cmap='RdYlGn',
    alpha=0.55, edgecolors='white', linewidths=0.3, s=45)
plt.colorbar(sc, ax=axes[0,0], label='Survived (1=Yes, 0=No)')
axes[0,0].set_title('Scatter: Age vs Fare — by Survival')
axes[0,0].set_xlabel('Age (years)')
axes[0,0].set_ylabel('Fare (£)')

# Chart 3-2: Scatter — Age vs Fare, coloured by Passenger Class
class_colors = {1: PALETTE[0], 2: PALETTE[2], 3: PALETTE[1]}
for cls, grp in df_raw.groupby('pclass'):
    axes[0,1].scatter(grp['age'],
                      grp['fare'].clip(upper=fare_99th_percentile),
                      color=class_colors[cls], alpha=0.45, s=40,
                      edgecolors='white', linewidths=0.3,
                      label=f'Class {cls}')
axes[0,1].set_title('Scatter: Age vs Fare — by Passenger Class')
axes[0,1].set_xlabel('Age (years)')
axes[0,1].set_ylabel('Fare (£)')
axes[0,1].legend()

# Chart 3-3: Violin Plot — Fare Per Person by Survival
# A violin plot shows the full distribution shape — wider = more passengers there
tmp = df.copy()
tmp['Outcome'] = df['survived'].map({0: 'Died', 1: 'Survived'})
sns.violinplot(data=tmp, x='Outcome', y='fare_per_person',
               ax=axes[0,2], palette=[PALETTE[1], PALETTE[2]], inner='box')
axes[0,2].set_title('Violin: Fare Per Person vs Survival')
axes[0,2].set_ylabel('Fare Per Person (£)')
axes[0,2].set_ylim(0, df['fare_per_person'].quantile(0.97))

# Chart 3-4: Grouped Bar — Survival Rate by Sex × Class
# Shows how gender interacts with class in determining survival
sex_cls = df_raw.groupby(['pclass', 'sex'])['survived'].mean().unstack() * 100
x, w = np.arange(3), 0.35
axes[1,0].bar(x-w/2, sex_cls['female'], w,
              label='Female', color=PALETTE[4], edgecolor='white')
axes[1,0].bar(x+w/2, sex_cls['male'],   w,
              label='Male',   color=PALETTE[0], edgecolor='white')
axes[1,0].set_title('Grouped Bar: Survival — Sex × Class')
axes[1,0].set_xticks(x)
axes[1,0].set_xticklabels(['1st Class', '2nd Class', '3rd Class'])
axes[1,0].set_ylabel('Survival Rate (%)')
axes[1,0].set_ylim(0, 115)
axes[1,0].legend()

# Chart 3-5: Grouped Bar — Survival by Embarkation × Sex
emb_sex = df_raw.groupby(['embarked', 'sex'])['survived'].mean().unstack() * 100
x2 = np.arange(len(emb_sex))
axes[1,1].bar(x2-w/2, emb_sex['female'], w,
              label='Female', color=PALETTE[4], edgecolor='white')
axes[1,1].bar(x2+w/2, emb_sex['male'],   w,
              label='Male',   color=PALETTE[0], edgecolor='white')
axes[1,1].set_title('Grouped Bar: Survival — Embarkation × Sex')
axes[1,1].set_xticks(x2)
axes[1,1].set_xticklabels(['Cherbourg', 'Queenstown', 'Southampton'])
axes[1,1].set_ylabel('Survival Rate (%)')
axes[1,1].set_ylim(0, 115)
axes[1,1].legend()

# Chart 3-6: KDE Density — Age by Survival
# Two overlapping density curves show where age distributions differ
for sv_val, lbl, col in [(0, 'Died', PALETTE[1]), (1, 'Survived', PALETTE[2])]:
    df[df['survived'] == sv_val]['age'].plot.kde(
        ax=axes[1,2], label=lbl, color=col, linewidth=2.5)
    line = axes[1,2].lines[-1]
    axes[1,2].fill_between(line.get_xdata(), line.get_ydata(),
                           alpha=0.12, color=col)
axes[1,2].set_title('KDE: Age Distribution — Survived vs Died')
axes[1,2].set_xlabel('Age (years)')
axes[1,2].set_ylabel('Density')
axes[1,2].set_xlim(0, 80)
axes[1,2].legend()

plt.tight_layout()
save_figure(fig, 'fig3_relationships.png')


# FIGURE 4 — PAIR PLOT (Multi-Variable Pairwise Relationships)

print('  Generating Figure 4: Pair Plot …')

# Select the most important numerical variables
pair_df = df[['survived', 'age', 'fare', 'family_size', 'fare_per_person']].copy()
pair_df['Survival'] = pair_df['survived'].map({0: 'Died', 1: 'Survived'})

g = sns.pairplot(
    pair_df.drop(columns='survived'),
    hue       = 'Survival',
    palette   = {'Died': PALETTE[1], 'Survived': PALETTE[2]},
    diag_kind = 'kde',          # KDE density on the diagonal
    plot_kws  = {'alpha': 0.45, 'edgecolor': 'none', 's': 25},
    diag_kws  = {'linewidth': 2.0}
)
g.figure.suptitle('Figure 4 — Pair Plot: All Pairwise Variable Relationships',
                  fontsize=14, fontweight='bold', y=1.02)
g.figure.set_facecolor(BG_COLOR)
save_figure(g.figure, 'fig4_pair_plot.png')


# FIGURE 5 — CORRELATION ANALYSIS (Heatmap + Ranked Bar)

print('  Generating Figure 5: Correlation Heatmap …')

fig, axes = plt.subplots(1, 2, figsize=(20, 9), facecolor=BG_COLOR)
fig.suptitle('Figure 5 — Correlation Analysis: Heatmap & Rankings vs Survived',
             fontsize=16, fontweight='bold', y=1.02)

# Select meaningful numerical features for the correlation matrix
corr_cols = ['survived', 'age', 'sibsp', 'parch', 'fare', 'family_size',
             'is_alone', 'sex_encoded', 'embarked_encoded',
             'fare_per_person', 'pclass_1', 'pclass_2', 'pclass_3']
corr_matrix = df[corr_cols].corr().round(3)

# Mask upper triangle — each pair appears only once (avoid redundancy)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask       = mask,
    annot      = True,          # Show r values inside each cell
    fmt        = '.2f',         # 2 decimal places
    cmap       = 'RdYlGn',      # Red=negative, Yellow=zero, Green=positive
    ax         = axes[0],
    linewidths = 0.5,
    linecolor  = 'white',
    cbar_kws   = {'shrink': 0.8, 'label': 'Pearson r'},
    vmin=-1, vmax=1,
    annot_kws  = {'size': 8}
)
axes[0].set_title('Full Correlation Heatmap (Lower Triangle)')
axes[0].tick_params(axis='x', rotation=35, labelsize=9)
axes[0].tick_params(axis='y', labelsize=9)

# Ranked bar chart — shows which features correlate most with survived
surv_corr = corr_matrix['survived'].drop('survived').sort_values()
bar_colors = [PALETTE[1] if v < 0 else PALETTE[2] for v in surv_corr.values]
axes[1].barh(surv_corr.index, surv_corr.values,
             color=bar_colors, edgecolor='white')
axes[1].axvline(0, color='black', linewidth=0.8)
for i, (idx, val) in enumerate(surv_corr.items()):
    offset = 0.01 if val >= 0 else -0.01
    align  = 'left' if val >= 0 else 'right'
    axes[1].text(val + offset, i, f'{val:+.3f}',
                 va='center', ha=align, fontsize=9)
axes[1].set_title('Features Ranked by Correlation with "Survived"')
axes[1].set_xlabel('Pearson Correlation Coefficient (r)')
axes[1].set_xlim(-0.6, 0.6)

plt.tight_layout()
save_figure(fig, 'fig5_correlation.png')


# FIGURE 6 — OUTLIER BOX PLOTS (Before & After Treatment)


print('  Generating Figure 6: Outlier Box Plots …')

fig, axes = plt.subplots(2, 4, figsize=(20, 11), facecolor=BG_COLOR)
fig.suptitle('Figure 6 — Outlier Detection & Treatment: Before vs After',
             fontsize=16, fontweight='bold', y=1.02)

# Shared box plot style across all charts
bp_style = dict(
    patch_artist = True,    # Fill boxes with colour
    notch        = False,
    medianprops  = dict(color='white', linewidth=2.5),
    whiskerprops = dict(linewidth=1.5),
    capprops     = dict(linewidth=1.5),
    flierprops   = dict(marker='o', markersize=4, alpha=0.5,
                        markerfacecolor=PALETTE[1], markeredgecolor='none')
)

variables = [
    ('fare_raw', 'fare',  'Fare — BEFORE', 'Fare — AFTER Capping', PALETTE[3]),
    ('age',      'age',   'Age',           'Age (No Change Needed)', PALETTE[0]),
    ('sibsp',    'sibsp', 'Siblings/Spouses', 'Siblings/Spouses',    PALETTE[2]),
    ('parch',    'parch', 'Parents/Children', 'Parents/Children',    PALETTE[4]),
]

for c, (raw_col, clean_col, title_before, title_after, color) in enumerate(variables):

    # Top row — raw data BEFORE treatment
    ax_top   = axes[0, c]
    raw_data = df[raw_col].dropna()
    bp_top   = ax_top.boxplot(raw_data, **bp_style)
    bp_top['boxes'][0].set_facecolor(color)
    bp_top['boxes'][0].set_alpha(0.75)
    s = raw_data.describe()
    ax_top.set_title(title_before)
    ax_top.set_ylabel('Value')
    ax_top.text(1.38, s['50%'],
                f"Median: {s['50%']:.1f}\nQ1: {s['25%']:.1f}\nQ3: {s['75%']:.1f}\nMax: {s['max']:.1f}",
                va='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    # Bottom row — cleaned data AFTER treatment
    ax_bot     = axes[1, c]
    clean_data = df[clean_col].dropna()
    bp_bot     = ax_bot.boxplot(clean_data, **bp_style)
    bp_bot['boxes'][0].set_facecolor(PALETTE[2])   # Green = cleaned/OK
    bp_bot['boxes'][0].set_alpha(0.75)
    s2 = clean_data.describe()
    ax_bot.set_title(title_after)
    ax_bot.set_ylabel('Value')
    ax_bot.text(1.38, s2['50%'],
                f"Median: {s2['50%']:.1f}\nQ1: {s2['25%']:.1f}\nQ3: {s2['75%']:.1f}\nMax: {s2['max']:.1f}",
                va='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

plt.tight_layout()
save_figure(fig, 'fig6_outlier_boxplots.png')



# FIGURE 7 — FEATURE ENGINEERING DEEP DIVE


print('  Generating Figure 7: Feature Engineering Deep Dive …')

fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor=BG_COLOR)
fig.suptitle('Figure 7 — Feature Engineering: New Features & Their Analytical Value',
             fontsize=16, fontweight='bold', y=1.02)

# Chart 7-1: fare_band — confirms bands correctly segment the fare range
for i, (band, grp) in enumerate(df.groupby('fare_band', observed=True)):
    axes[0,0].scatter(grp.index, grp['fare'],
                      color=PALETTE[i], alpha=0.5, s=20, label=str(band))
axes[0,0].set_title('fare_band — Fare Range Segmentation')
axes[0,0].set_xlabel('Passenger Index')
axes[0,0].set_ylabel('Fare (£)')
axes[0,0].legend(title='Fare Band', fontsize=8)

# Chart 7-2: fare_per_person by class — violin plot
fare_cls = df_raw.assign(fare_per_person=df['fare_per_person'])
sns.violinplot(data=fare_cls, x='pclass', y='fare_per_person',
               ax=axes[0,1], palette=PALETTE[:3], inner='quartile')
axes[0,1].set_title('fare_per_person — By Passenger Class')
axes[0,1].set_xlabel('Passenger Class')
axes[0,1].set_ylabel('Fare Per Person (£)')
axes[0,1].set_ylim(0, df['fare_per_person'].quantile(0.97))

# Chart 7-3: age_group × sex — stacked bar shows passenger composition
ag_sex = df.groupby(['age_group', 'sex'], observed=True).size().unstack(fill_value=0)
ag_sex.plot(kind='bar', ax=axes[0,2], stacked=True,
            color=[PALETTE[4], PALETTE[0]], edgecolor='white')
axes[0,2].set_title('age_group × sex — Passenger Composition')
axes[0,2].set_xlabel('Age Group')
axes[0,2].set_ylabel('Passenger Count')
axes[0,2].tick_params(axis='x', rotation=15)
axes[0,2].legend(title='Sex', labels=['Female', 'Male'])

# Chart 7-4: is_alone and family_size vs survival (jittered scatter)
# Jitter (small random noise) prevents dots from overlapping on 0/1 axis
alone_colors = df['is_alone'].map({0: PALETTE[2], 1: PALETTE[1]})
axes[1,0].scatter(
    df['family_size'],
    df['survived'] + np.random.normal(0, 0.03, len(df)),
    c=alone_colors, alpha=0.4, s=30, edgecolors='none'
)
p1 = mpatches.Patch(color=PALETTE[2], label='With Family (is_alone=0)')
p2 = mpatches.Patch(color=PALETTE[1], label='Alone (is_alone=1)')
axes[1,0].legend(handles=[p1, p2], fontsize=8)
axes[1,0].set_title('is_alone + family_size — Effect on Survival')
axes[1,0].set_xlabel('Family Size')
axes[1,0].set_ylabel('Survived (0=Died, 1=Survived)')
axes[1,0].set_yticks([0, 1])
axes[1,0].set_yticklabels(['Died', 'Survived'])

# Chart 7-5: title — dual-axis chart (survival rate + passenger count)
title_agg = df.groupby('title')['survived'].agg(['mean', 'count'])
ax_r = axes[1,1].twinx()
bars = axes[1,1].bar(title_agg.index, title_agg['mean']*100,
                     color=PALETTE[3:6], edgecolor='white', width=0.5, alpha=0.85)
ax_r.plot(title_agg.index, title_agg['count'], 'D--',
          color='#333', linewidth=2, markersize=8, label='Count')
for bar, v in zip(bars, title_agg['mean']*100):
    axes[1,1].text(bar.get_x()+bar.get_width()/2, v+1,
                   f'{v:.0f}%', ha='center', fontsize=11, fontweight='bold')
axes[1,1].set_title('title — Survival Rate & Count per Title')
axes[1,1].set_ylabel('Survival Rate (%)')
ax_r.set_ylabel('Passenger Count')
ax_r.legend(fontsize=8)

# Chart 7-6: All scaled features on the same [0,1] axis
# Proves scaling worked — all features now share the same range
for col, lbl, color in [
    ('age_scaled',   'Age (scaled)',   PALETTE[0]),
    ('fare_scaled',  'Fare (scaled)',  PALETTE[3]),
    ('sibsp_scaled', 'Sibsp (scaled)', PALETTE[2]),
    ('parch_scaled', 'Parch (scaled)', PALETTE[4]),
]:
    if col in df.columns:
        df[col].plot.kde(ax=axes[1,2], label=lbl, color=color, linewidth=2.5)
axes[1,2].set_title('*_scaled — All Scaled Features on [0, 1] Range')
axes[1,2].set_xlabel('Scaled Value (0 = minimum, 1 = maximum)')
axes[1,2].set_ylabel('Density')
axes[1,2].legend()
axes[1,2].set_xlim(0, 1)

plt.tight_layout()
save_figure(fig, 'fig7_feature_engineering.png')


# SECTION 7 — STEP 3C: KEY INSIGHTS SUMMARY
# GOAL: Turn the EDA findings into clear, numbered, actionable insights.


print_banner('STEP 3C — KEY INSIGHTS SUMMARY')

# Pre-compute all statistics used in the insights report
total        = len(df)
surv_n       = df['survived'].sum()
surv_pct     = surv_n / total * 100
f_surv       = df[df['sex']=='female']['survived'].mean() * 100
m_surv       = df[df['sex']=='male']['survived'].mean() * 100
c1_surv      = df_raw[df_raw['pclass']==1]['survived'].mean() * 100
c2_surv      = df_raw[df_raw['pclass']==2]['survived'].mean() * 100
c3_surv      = df_raw[df_raw['pclass']==3]['survived'].mean() * 100
child_surv   = df[df['age_group']=='Child']['survived'].mean() * 100
senior_surv  = df[df['age_group']=='Senior']['survived'].mean() * 100
alone_surv   = df[df['is_alone']==1]['survived'].mean() * 100
family_surv  = df[df['is_alone']==0]['survived'].mean() * 100
cherb_surv   = df[df['embark_town']=='Cherbourg']['survived'].mean() * 100
south_surv   = df[df['embark_town']=='Southampton']['survived'].mean() * 100
prem_surv    = df[df['fare_band']=='Premium']['survived'].mean() * 100
budg_surv    = df[df['fare_band']=='Budget']['survived'].mean() * 100
sex_r        = df['sex_encoded'].corr(df['survived'])
fare_r       = df['fare'].corr(df['survived'])

INSIGHTS = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              KEY INSIGHTS — TITANIC SURVIVAL ANALYSIS                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

  DATASET OVERVIEW
  ──────────────────────────────────────────────────────────────────────────────
  • Total Passengers  : {total:,}
  • Survived          : {surv_n} ({surv_pct:.1f}%)
  • Did Not Survive   : {total - surv_n} ({100 - surv_pct:.1f}%)
  • Raw Columns       : 12
  • After Engineering : {df.shape[1]} columns


  INSIGHT 1 — GENDER IS THE STRONGEST PREDICTOR OF SURVIVAL
  ──────────────────────────────────────────────────────────────────────────────
  • Female survival rate      : {f_surv:.1f}%
  • Male survival rate        : {m_surv:.1f}%
  • Gap                       : {f_surv - m_surv:.1f} percentage points
  • Pearson r with survived   : {sex_r:.3f}  ← strongest correlation of all

  What this means:
  The "Women and children first" maritime evacuation policy is clearly visible
  in the data. Sex is by far the single strongest predictor of survival.
  Any predictive model MUST include sex as a feature.


  INSIGHT 2 — PASSENGER CLASS REVEALS STARK SOCIAL INEQUALITY
  ──────────────────────────────────────────────────────────────────────────────
  • 1st class survival : {c1_surv:.1f}%
  • 2nd class survival : {c2_surv:.1f}%
  • 3rd class survival : {c3_surv:.1f}%
  • 1st vs 3rd ratio   : {c1_surv/c3_surv:.1f}× more likely to survive

  What this means:
  Passenger class influenced cabin location (upper decks = closer to lifeboats),
  access to safety information, and priority treatment during evacuation.
  Wealth and social status had a direct impact on who lived and who died.


  INSIGHT 3 — CHILDREN WERE CLEARLY PRIORITISED IN EVACUATION
  ──────────────────────────────────────────────────────────────────────────────
  • Children (0–12)  survival : {child_surv:.1f}%   ← highest of all age groups
  • Seniors  (60+)   survival : {senior_surv:.1f}%

  What this means:
  The evacuation policy explicitly prioritised children. They had the highest
  survival rate of any age group, consistent with historical records.


  INSIGHT 4 — FARE IS A STRONG PROXY FOR WEALTH AND CLASS
  ──────────────────────────────────────────────────────────────────────────────
  • Premium fare band survival : {prem_surv:.1f}%
  • Budget  fare band survival : {budg_surv:.1f}%
  • Pearson r with survived    : {fare_r:+.3f}

  What this means:
  Fare acts as a continuous measure of class and wealth beyond the 3-tier pclass.
  Passengers who paid more survived at significantly higher rates.
  The engineered fare_per_person feature refines this even further.


  INSIGHT 5 — EMBARKATION PORT REFLECTS CLASS COMPOSITION
  ──────────────────────────────────────────────────────────────────────────────
  • Cherbourg (C)   survival : {cherb_surv:.1f}%
  • Southampton (S) survival : {south_surv:.1f}%

  What this means:
  Cherbourg was primarily a boarding point for wealthy 1st-class passengers,
  explaining its higher survival rate. Southampton was the main boarding point
  for 3rd-class passengers, resulting in a lower overall survival rate.
  Embarkation port is a CLASS PROXY, not a direct causal factor.


  INSIGHT 6 — FAMILY SIZE HAS A NON-LINEAR EFFECT ON SURVIVAL
  ──────────────────────────────────────────────────────────────────────────────
  • Travelling alone       : {alone_surv:.1f}% survival rate
  • Travelling with family : {family_surv:.1f}% survival rate

  What this means:
  Small families (2–4 members) had the best survival odds — enough people to
  help each other, but small enough to move quickly. Very large families (7+)
  had the worst odds — difficult to keep the group together during evacuation.
  Solo travellers had no family support during the crisis.


  INSIGHT 7 — DATA QUALITY FINDINGS & DECISIONS
  ──────────────────────────────────────────────────────────────────────────────
  Problem              Action Taken                      Reason
  ───────────────────  ────────────────────────────────  ─────────────────────
  age: 19.4% missing   Grouped median imputation          Respects subgroups
  deck: 76.5% missing  Column dropped entirely            Too sparse to impute
  fare: outlier £512   Capped at 99th pct (£{fare_99th_percentile:.0f})         Reduce distortion
  sibsp/parch: high    Retained as-is                     Real large families


  INSIGHT 8 — TOP FEATURES FOR PREDICTIVE MODELLING
  ──────────────────────────────────────────────────────────────────────────────
  Ranked by Pearson correlation with survived:

    Rank  Feature              r value   Interpretation
    ────  ───────────────────  ────────  ──────────────────────────────────────
      1   sex_encoded          {df['sex_encoded'].corr(df['survived']):+.3f}    Gender — strongest signal
      2   pclass_1             {df['pclass_1'].corr(df['survived']):+.3f}    Being 1st class = higher survival
      3   fare                 {df['fare'].corr(df['survived']):+.3f}    Wealth proxy
      4   fare_per_person      {df['fare_per_person'].corr(df['survived']):+.3f}    Refined wealth indicator
      5   family_size          {df['family_size'].corr(df['survived']):+.3f}    Group size effect
      6   age                  {df['age'].corr(df['survived']):+.3f}    Age-based priority
      7   is_alone             {df['is_alone'].corr(df['survived']):+.3f}    Solo traveller disadvantage

  

══════════════════════════════════════════════════════════════════════════════════
"""

print(INSIGHTS)

# Save to file
with open(f'{OUTPUT_DIR}/key_insights.txt', 'w') as f:
    f.write(INSIGHTS)
print(f'   Key insights saved → outputs/key_insights.txt')


#  Final Summary 
print_banner('PIPELINE COMPLETE')
print(f"""
    Step 1 — Data Loading & Exploration      DONE
    Step 2 — Data Cleaning & Transformation  DONE
    Step 3A — Descriptive Statistics         DONE
    Step 3B — 7 Visualization Figures        DONE
    Step 3C — Key Insights (8 insights)      DONE

 
""")
