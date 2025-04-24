# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # German Credit Risk Classification in Databricks
# MAGIC This notebook demonstrates:
# MAGIC 1. Reading data from DBFS via Spark
# MAGIC 2. Converting to pandas for EDA
# MAGIC 3. Parameterizing with widgets
# MAGIC 4. Using MLflow for experiment tracking
# MAGIC 5. Persisting cleaned data as a Delta table
# MAGIC 6. Full pandas/​scikit-learn workflow (EDA → preprocessing → modeling)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 1. Setup & Imports
# MAGIC

# COMMAND ----------

# MAGIC %pip uninstall -y mlflow protobuf typing_extensions pydantic
# MAGIC %pip install typing_extensions==4.8.0
# MAGIC %pip install pydantic==1.10.13
# MAGIC %pip install protobuf==4.25.1
# MAGIC %pip install mlflow==2.9.2

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix,
    precision_recall_curve, auc
)

from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE, KMeansSMOTE

import mlflow
import mlflow.sklearn

# Enable MLflow autologging for sklearn
# mlflow.sklearn.autolog(disable=False)

sns.set_theme(style="whitegrid")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 2. Widgets for Parameterization
# MAGIC

# COMMAND ----------

dbutils.widgets.text("train_fraction", "0.8", "Train / Test Split Fraction")
dbutils.widgets.text("random_seed", "42", "Random Seed")
train_frac = float(dbutils.widgets.get("train_fraction"))
seed       = int(dbutils.widgets.get("random_seed"))

print(f"Train/Test split: {train_frac:.2f}/{1-train_frac:.2f}, seed={seed}")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 3. Load Processed Data via Spark
# MAGIC

# COMMAND ----------

df_spark = spark.table("credit_features")
display(df_spark.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 4. Rename Columns
# MAGIC

# COMMAND ----------

col_map_df = spark.table("workspace.default.col_mapping")
display(col_map_df.limit(5))

# COMMAND ----------

# Load the table
col_map_df = spark.table("workspace.default.col_mapping")
# Convert to pandas
col_map_pd = col_map_df.toPandas()

col_map = col_map_pd.iloc[0].to_dict()
df_spark = df_spark.select([col(c).alias(col_map.get(c, c)) for c in df_spark.columns])

# Confirm
print("✅ Columns after renaming:")
print(df_spark.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 5. Persist Cleaned Data as Delta & Register Table
# MAGIC

# COMMAND ----------

# df_spark.write \
#     .format("delta") \
#     .mode("overwrite") \
#     .saveAsTable("german_credit_features")

# print("✅ Table `german_credit_features` saved as a managed Delta table.")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 6. Convert to pandas for EDA & Modelling
# MAGIC

# COMMAND ----------

df = df_spark.toPandas()
print(f"Loaded pandas DataFrame with shape {df.shape}")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 7. Create Target & Basic Cleaning
# MAGIC

# COMMAND ----------

# Map Risk Label → binary Target
df['Target'] = df['Risk Label'].map({1: 1, 2: 0})

# Fill NA for key categoricals
cat_fill = [
  'Checking Account Status','Savings Account','Credit History',
  'Loan Purpose','Housing Type','Employment Length',
  'Other Debtors','Property Type','Installment Plans',
  'Telephone Status','Foreign Worker','Personal Status & Sex'
]
for c in cat_fill:
    if c in df:
        df[c] = df[c].fillna('Unknown')

# Binary encode Sex from Personal Status & Sex
df['Sex'] = df['Personal Status & Sex'].apply(lambda x: 1 if x.lower().startswith('m') else 0)

display(df[['Risk Label','Target','Sex']].head())


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 8. Descriptive Statistics
# MAGIC

# COMMAND ----------

num_df = df.select_dtypes(include='number').drop(columns=['Risk Label','Target'], errors='ignore')
display(num_df.describe().round(2))


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 9. Class Balance
# MAGIC

# COMMAND ----------

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# Count plot
ax = sns.countplot(
    x='Risk Label', hue='Risk Label', data=df,
    palette=['#77DD76', '#FF6962'], ax=axes[0]
)
ax.set_xticks([0, 1])  # Set tick positions
ax.set_xticklabels(['Good (1)', 'Bad (2)'])  # Set tick labels
ax.set_title('Credit Risk Distribution')
for p in ax.patches:
    ax.text(
        p.get_x() + p.get_width() / 2,
        p.get_height() + 5,
        p.get_height(),
        ha='center'
    )
ax.set_ylabel('Count')
ax.set_xlabel('Risk Label')

# Pie chart
risk_counts = df['Risk Label'].value_counts()
labels = ['Good (1)', 'Bad (2)']
colors = ['#77DD76', '#FF6962']
axes[1].pie(
    risk_counts, labels=labels, autopct='%1.1f%%',
    startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'}
)
axes[1].set_title('Credit Risk Distribution')

# Adjust layout
plt.tight_layout()
#plt.savefig("plots/credit_eda.svg")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 10. Numerical Distributions by Risk
# MAGIC

# COMMAND ----------

df['Risk Label Mapped'] = df['Risk Label'].map({1: 'Good', 2: 'Bad'})
# # Check unique values in the 'Risk Label' column
# print("Unique values in 'Risk Label':", df['Risk Label'].unique())
#
# # Check the value counts in the 'Risk Label' column
# print("Value counts in 'Risk Label':")
# print(df['Risk Label'].value_counts())

num_feats = ['Credit Amount (DM)', 'Loan Duration (Months)', 'Age (Years)']

# Create a single row of subplots
fig, axes = plt.subplots(1, len(num_feats), figsize=(18, 5), sharey=False)

for ax, feat in zip(axes, num_feats):
    sns.kdeplot(
        data=df, x=feat, hue='Risk Label Mapped',
        fill=True, common_norm=False,
        palette=['#77DD76', '#FF6962'], ax=ax
    )
    ax.set_title(f"{feat} Distribution by Risk")
    ax.set_xlabel(feat)
    ax.set_ylabel('Density')

plt.tight_layout()
plots_path = "plots/numerical_distribution.svg"
#plt.savefig(plots_path)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 11. Categorical Distributions by Risk
# MAGIC

# COMMAND ----------

# Map for value translations on final column names
value_maps = {
    'Loan Purpose': {
        'car_new': 'Car (New)',
        'car_used': 'Car (Used)',
        'furniture': 'Furniture',
        'radio_tv': 'Radio/TV',
        'domestic_app': 'Domestic Appliances',
        'repairs': 'Repairs',
        'education': 'Education',
        'vacation': 'Vacation',
        'retraining': 'Retraining',
        'business': 'Business',
        'others': 'Other'
    },
    'Housing Type': {
        'rent': 'Rented',
        'own': 'Owned',
        'free': 'Free'
    },
    'Personal Status & Sex': {
        'm_div_sep': 'Male Divorced/Separated',
        'f_div_sep_mar': 'Female Divorced/Married',
        'm_single': 'Male Single',
        'm_mar_wid': 'Male Married/Widowed',
        'f_single': 'Female Single'
    }
}

# Now you don’t need display_to_actual anymore!
cat_feats = ['Loan Purpose', 'Housing Type', 'Personal Status & Sex']

for col in cat_feats:
    df_plot = df.copy()

    # Map column values if applicable
    if col in value_maps:
        df_plot[col] = df_plot[col].map(value_maps[col])

    # Define a consistent category order
    category_order = df_plot[col].value_counts().sort_index().index.tolist()

    # Compute percentage-based DataFrame
    crosstab = pd.crosstab(df_plot[col], df_plot['Risk Label Mapped'], normalize='index') * 100
    crosstab = crosstab[['Good', 'Bad']]  # Ensure correct column order
    crosstab = crosstab.reindex(category_order)  # Reindex rows to match category order

    # Create side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw count plot
    sns.countplot(
        data=df_plot, x=col, hue='Risk Label Mapped',
        palette=['#77DD76', '#FF6962'],
        order=category_order,
        ax=axes[0]
    )
    axes[0].set_title(f"{col} by Risk (Count)")
    axes[0].tick_params(axis='x', rotation=45)

    # Percentage plot
    crosstab.plot(
        kind='bar', stacked=True,
        color=['#77DD76', '#FF6962'],
        ax=axes[1]
    )
    axes[1].set_title(f"{col} by Risk (%)")
    axes[1].set_ylabel("Percentage")
    axes[1].legend(title='Risk Label')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plots_path = f"plots/categorical_distribution_{col}.svg"
    #plt.savefig(plots_path)
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 12. Correlation Heatmap (Top-20 Features)
# MAGIC

# COMMAND ----------

from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

# Create target column
df['Target'] = df['Risk Label'].map({1: 1, 2: 0})

# Encode categorical features
enc = pd.get_dummies(df.drop(columns=['Risk Label', 'Target']), drop_first=True)
enc['Target'] = df['Target']

# Remove specific one-hot encoded columns
cols_to_remove = [
    'Age (Years)',
    'No Credits',
    'Sex',
    'Male Divorced/Separated',
    'Female Divorced/Married',
    'Male Single',
]
enc.drop(columns=[col for col in cols_to_remove if col in enc.columns], inplace=True)
# Select only numeric columns
enc_numeric = enc.select_dtypes(include=['number'])

# 1. Mapping column to display name (as given)
col_display_names = {
    "chk_lt_0": "Checking <0 DM",
    "chk_0_200": "Checking 0–200 DM",
    "chk_ge_200": "Checking ≥200 DM",
    "chk_none": "No Checking Account",
    "hist_no_credits": "No Credits",
    "hist_all_paid": "All Paid Duly",
    "hist_existing_paid": "Existing Paid Duly",
    "hist_past_delay": "Past Delay",
    "hist_critical": "Critical History",
    "pur_car_new": "Purpose: New Car",
    "pur_car_used": "Purpose: Used Car",
    "pur_furniture": "Purpose: Furniture",
    "pur_radio_tv": "Purpose: Radio/TV",
    "pur_domestic": "Purpose: Domestic Appliances",
    "pur_repairs": "Purpose: Repairs",
    "pur_education": "Purpose: Education",
    "pur_vacation": "Purpose: Vacation",
    "pur_retraining": "Purpose: Retraining",
    "pur_business": "Purpose: Business",
    "pur_others": "Purpose: Others",
    "sav_lt_100": "Savings <100 DM",
    "sav_100_500": "Savings 100–500 DM",
    "sav_500_1000": "Savings 500–1000 DM",
    "sav_ge_1000": "Savings ≥1000 DM",
    "sav_unknown": "Savings Unknown",
    "emp_unempl": "Unemployed",
    "emp_lt1": "Employed <1yr",
    "emp_1_4": "Employed 1–4yrs",
    "emp_4_7": "Employed 4–7yrs",
    "emp_ge7": "Employed ≥7yrs",
    "sex_m_divsep": "Male Divorced/Separated",
    "sex_f_divsep_mar": "Female Divorced/Married",
    "sex_m_single": "Male Single",
    "sex_m_marwid": "Male Married/Widowed",
    "sex_f_single": "Female Single",
    "debt_none": "No Debtors",
    "debt_coapp": "Co-applicant",
    "debt_guarantor": "Guarantor",
    "prop_real": "Property: Real Estate",
    "prop_sav_ins": "Property: Savings Insurance",
    "prop_car_other": "Property: Car/Other",
    "prop_unknown": "Property: Unknown",
    "inst_bank": "Installment: Bank",
    "inst_stores": "Installment: Stores",
    "inst_none": "Installment: None",
    "house_rent": "Rented",
    "house_own": "Owned",
    "house_free": "Free Housing",
    "job_unsk_nonres": "Unskilled (Non-resident)",
    "job_unsk_res": "Unskilled (Resident)",
    "job_skilled": "Skilled Worker",
    "job_manage": "Management",
    "tel_none": "No Telephone",
    "tel_yes": "Telephone Registered",
    "foreign_yes": "Foreign Worker (Yes)",
    "foreign_no": "Foreign Worker (No)"
}

# 2. Manually assign group names (you can reorder these)
group_keywords = {
    'Checking': ['chk_'],
    'Credit History': ['hist_'],
    'Purpose': ['pur_'],
    'Savings': ['sav_'],
    'Employment': ['emp_'],
    'Sex': ['Personal Status & Sex'],
    'Debtors': ['debt_'],
    'Property': ['prop_'],
    'Installment Plans': ['inst_'],
    'Housing': ['house_'],
    'Job': ['job_'],
    'Telephone': ['tel_'],
    'Foreign Worker': ['foreign_'],
}

# Explicit fallback for misclassified features
explicit_group_map = {
    'chk_none': 'Checking',
    'chk_lt_0': 'Checking',
    'chk_0_200': 'Checking',
    'chk_ge_200': 'Checking',
}

# 3. Reverse-assign groups with fallback
grouped_cols = defaultdict(list)

for col in enc.columns:
    if col == "Target":
        continue

    # First check explicit mapping
    if col in explicit_group_map:
        grouped_cols[explicit_group_map[col]].append(col)
        continue

    # Match by any prefix in group_keywords
    assigned = False
    for group, prefixes in group_keywords.items():
        for prefix in prefixes:
            if col.startswith(prefix):  # more robust for multiple prefixes
                grouped_cols[group].append(col)
                assigned = True
                break
        if assigned:
            break

    if not assigned:
        grouped_cols["Other"].append(col)

# 4. Sort columns by group order
group_order = list(group_keywords.keys()) + ['Other']
ordered_columns = []
for group in group_order:
    ordered_columns.extend(sorted(grouped_cols[group]))

# 5. Filter numeric and reorder
enc_numeric_ordered = enc[ordered_columns + ['Target']].select_dtypes(include='number')

# 6. Pick top 20 by absolute correlation to Target, but preserve group order
corr_to_target = enc_numeric_ordered.corr()['Target'].abs().sort_values(ascending=False).index[1:31]
top20_cols = [col for col in ordered_columns if col in corr_to_target]

# 7. Plot heatmap using ordered top 20
sub = enc_numeric_ordered[top20_cols]

rcParams['font.family'] = 'DejaVu Sans'
plt.figure(figsize=(40, 30))
sns.heatmap(
    sub.corr(), cmap='RdBu_r', center=0,
    annot=True, fmt=".2f", linewidths=0.5,
    xticklabels=sub.columns, yticklabels=sub.columns
)
plt.title("Top‑20 Feature Correlations (Grouped by Logic)")
plt.tight_layout()
plots_path = "plots/corrlation_heatmaps.svg"
#plt.savefig(plots_path)
plt.show()

# print([col for col in enc.columns if 'sex' in col.lower()])

# # Debug print to verify which columns were grouped
# for group, cols in grouped_cols.items():
#     print(f"\nGroup: {group} ({len(cols)} features)")
#     for col in cols:
#         print(f"  - {col}")

# COMMAND ----------

# MAGIC %md
# MAGIC 13. Normality Check (Q–Q Plots)

# COMMAND ----------

from scipy import stats

num_feats = ['Credit Amount (DM)', 'Loan Duration (Months)', 'Age (Years)']
fig, axes = plt.subplots(1, len(num_feats), figsize=(5*len(num_feats), 4))
for ax, feat in zip(axes, num_feats):
    stats.probplot(df[feat], dist='norm', plot=ax)
    ax.set_title(f"Q–Q: {feat}")
plt.tight_layout()
plots_path = "plots/qq_plots.svg"
#plt.savefig(plots_path)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### 14. Boxplot: Loan Amount by Risk
# MAGIC

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Boxplot 1: Credit Amount
sns.boxplot(
    x='Risk Label', y='Credit Amount (DM)', hue='Risk Label',
    data=df, palette=['#77DD76', '#FF6962'], dodge=False, ax=axes[0]
)
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Good (1)', 'Bad (2)'])
axes[0].set_title('Loan Amount by Risk')
axes[0].legend_.remove()  # Remove legend from first plot

# Boxplot 2: Loan Duration
sns.boxplot(
    x='Target', y='Loan Duration (Months)', hue='Target',
    data=df, palette=['#FF6962', '#77DD76'], ax=axes[1]
)
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['Bad', 'Good'])
axes[1].set_title('Loan Duration by Risk')
axes[1].set_xlabel('Credit Risk')
axes[1].set_ylabel('Loan Duration (Months)')
axes[1].legend_.remove()

# Boxplot 3: Age
sns.boxplot(
    x='Target', y='Age (Years)', hue='Target',
    data=df, palette=['#FF6962', '#77DD76'], ax=axes[2]
)
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels(['Bad', 'Good'])
axes[2].set_title('Age by Risk')
axes[2].set_xlabel('Credit Risk')
axes[2].set_ylabel('Age (Years)')
axes[2].legend_.remove()

plt.tight_layout()
plots_path = "plots/box_plots.svg"
#plt.savefig(plots_path)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 15. Train/Test Split & Scaling
# MAGIC

# COMMAND ----------

# Fill NA values with 'Unknown' for certain categorical features
cat_fill_cols = [
    'Checking Account Status', 'Savings Account',
    'Credit History', 'Loan Purpose', 'Housing Type',
    'Employment Length', 'Other Debtors', 'Property Type',
    'Installment Plans', 'Telephone Status', 'Foreign Worker',
    'Personal Status & Sex'
]
for col in cat_fill_cols:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

# Binary encode 'Sex' from 'Personal Status & Sex'
# df['Sex'] = df['Personal Status & Sex'].apply(lambda x: 'male' in x.lower()).astype(int)
df['Sex'] = df['Personal Status & Sex'].apply(lambda x: 1 if x[0].lower() == 'm' else 0)


# Encode target: 1 = Good, 0 = Bad
df['Target'] = df['Risk Label'].map({1: 1, 2: 0})

# # One-hot encode all nominal/ordinal categoricals (except Risk Label and original codes)
df_encoded = pd.get_dummies(
    # df.drop(columns=['Risk Label', 'Personal Status & Sex']),
    df,
    drop_first=True
)

print(f"Encoded dataset shape: {df_encoded.shape}")

#

# COMMAND ----------

X = df_encoded.drop(columns=['Target'])
y = df_encoded['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_frac, stratify=y, random_state=seed
)

std = StandardScaler().fit(X_train)
X_train_std = pd.DataFrame(std.transform(X_train), columns=X_train.columns)
X_test_std  = pd.DataFrame(std.transform(X_test),  columns=X_test.columns)

print("Train/Test shapes:", X_train.shape, X_test.shape)
print("Class distribution (train):", y_train.value_counts(normalize=True).to_dict())


# COMMAND ----------

# MAGIC %md
# MAGIC ##16. Feature Scaling

# COMMAND ----------

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization (mean=0, std=1) on full feature set
std_scaler = StandardScaler()
X_train_std = pd.DataFrame(std_scaler.fit_transform(X_train), columns=X_train.columns)
X_test_std  = pd.DataFrame(std_scaler.transform(X_test), columns=X_test.columns)

# Min-Max Scaling (0–1) as alternative
mm_scaler = MinMaxScaler()
X_train_mm = pd.DataFrame(mm_scaler.fit_transform(X_train), columns=X_train.columns)
X_test_mm  = pd.DataFrame(mm_scaler.transform(X_test), columns=X_test.columns)

print("Standardization and Min-Max scaling complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 17. PCA (95% Variance)
# MAGIC

# COMMAND ----------

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# PCA on standardized full feature set
pca_temp = PCA()
X_pca_temp = pca_temp.fit_transform(X_train_std)
explained = pca_temp.explained_variance_ratio_

# Cumulative variance with strict monotonicity fix
cumvar = np.cumsum(explained)
epsilon = 1e-10
cumvar_strict = cumvar + epsilon * np.arange(len(cumvar))
cumvar_unique, idx_unique = np.unique(cumvar_strict, return_index=True)
explained_unique = explained[idx_unique]

# Scree plot
plt.figure(figsize=(6,4))
plt.plot(explained_unique, marker='o', label='Per PC')
plt.plot(cumvar_unique, marker='o', label='Cumulative')
plt.axhline(0.95, linestyle='--', color='gray')
plt.xlabel('Principal Component Index')
plt.ylabel('Explained Variance')
plt.title('PCA Explained Variance')
plt.legend()
plt.tight_layout()
plots_path = "plots/pca_explained_variance.svg"
#plt.savefig(plots_path)
plt.show()

# Select number of components for 95% variance
n_components = np.argmax(cumvar_unique >= 0.95) + 1
print(f"Number of PCs explaining ≥95% variance: {n_components}")

# Final PCA transformation
pca_final = PCA(n_components=n_components)
X_train_pca = pca_final.fit_transform(X_train_std)
X_test_pca  = pca_final.transform(X_test_std)
print('PCA transformation complete.')

# Optional 2D Projection for visualization
pca_2d = PCA(n_components=2)
X_train_2d = pca_2d.fit_transform(X_train_std)

plt.figure(figsize=(6, 5))
sns.scatterplot(
    x=X_train_2d[:, 0], y=X_train_2d[:, 1],
    hue=y_train.map({1: 'Good', 0: 'Bad'}),
    palette=['#FF6962', '#77DD76'], alpha=0.7
)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection (Train Data)')
plt.legend(title='Credit Risk')
plt.tight_layout()
plots_path = "plots/pca_explained_variance_2d.svg"
#plt.savefig(plots_path)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 15. Addressing Class Imbalance
# MAGIC

# COMMAND ----------

from collections import Counter
fig,axes = plt.subplots(2,2,figsize=(10,8)); axes=axes.flatten()

def plot_cnt(ax,y,title):
    cnt=Counter(y); ax.bar(cnt.keys(),cnt.values(), color="#77DD76")
    ax.set_title(title); ax.set_ylim(0, max(cnt.values())*1.1)

plot_cnt(axes[0], y_train,        "Original")
ccX, ccy = ClusterCentroids(random_state=seed).fit_resample(X_train_pca, y_train)
plot_cnt(axes[1], ccy,            "ClusterCentroids")
smX, smy = SMOTE(random_state=seed).fit_resample(X_train_pca, y_train)
plot_cnt(axes[2], smy,            "SMOTE")
kmX, kmy = KMeansSMOTE(random_state=seed).fit_resample(X_train_std, y_train)
plot_cnt(axes[3], kmy,            "KMeans-SMOTE")

plt.tight_layout()
display(plt.gcf())


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 16. Classification Models w/ MLflow Tracking
# MAGIC

# COMMAND ----------

models = {
  "LogisticRegression": LogisticRegression(max_iter=1000, random_state=seed),
  "RandomForest":       RandomForestClassifier(random_state=seed),
  "SVM":                SVC(probability=True, random_state=seed)
}
param_grids = {
  "LogisticRegression": {"C":[0.01,0.1,1,10]},
  "RandomForest":       {"n_estimators":[50,100],"max_depth":[None,10,20]},
  "SVM":                {"C":[0.1,1,10],"kernel":["rbf","linear"]}
}

results = {}
# Comment MLflow parts if you don't need the registry
# mlflow.set_experiment("/Users/you@example.com/my_experiment")

for name, est in models.items():
    # Skip mlflow.start_run if causing issues
    print(f"Running: {name}")
    grid = GridSearchCV(est, param_grids[name], scoring='f1', cv=5, n_jobs=-1)
    grid.fit(X_train_pca, y_train)
    best = grid.best_estimator_
    y_pred = best.predict(X_test_pca)
    y_proba = best.predict_proba(X_test_pca)[:, 1]
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    pr, rc, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(rc, pr)

    results[name] = {
        "est": best,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "pr_auc": pr_auc
    }

    print(f"{name} → f1: {f1:.3f}, precision: {prec:.3f}, recall: {rec:.3f}, pr_auc: {pr_auc:.3f}")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 17. Results Visualization
# MAGIC

# COMMAND ----------

# F1 Bar
names = list(results.keys())
scores = [results[n]['f1'] for n in names]
fig,ax = plt.subplots(figsize=(6,4))
sns.barplot(x=scores, y=names, palette='Blues_r', dodge=False, ax=ax)
ax.set_xlim(0,1); ax.set_xlabel("Test F1"); ax.set_title("Model F1 Comparison")
plt.tight_layout()
display(plt.gcf())


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 18. Confusion Matrices & PR Curves
# MAGIC

# COMMAND ----------

fig,axes = plt.subplots(len(results),2,figsize=(10,4*len(results)))
for i,(name,res) in enumerate(results.items()):
    est = res['est']
    cm = confusion_matrix(y_test, est.predict(X_test_pca))
    sns.heatmap(cm, annot=True, fmt="d", ax=axes[i,0], cmap='Greens')
    axes[i,0].set_title(f"{name} Confusion")
    pr, rc, _ = precision_recall_curve(y_test, est.predict_proba(X_test_pca)[:,1])
    axes[i,1].plot(rc,pr,label=f"AUC={res['pr_auc']:.2f}")
    axes[i,1].hlines(y=y_test.mean(), xmin=0, xmax=1, linestyles='--', label="No Skill")
    axes[i,1].set_title(f"{name} PR Curve"); axes[i,1].legend()
plt.tight_layout()
display(plt.gcf())
