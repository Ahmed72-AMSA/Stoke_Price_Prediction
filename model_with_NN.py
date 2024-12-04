import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px  # This is helpful for easier rendering
import plotly.io as pio  # Use this for renderer configuration

# Set the renderer for Jupyter Notebooks
pio.renderers.default = 'notebook'  # This ensures the chart shows up within the notebook

# Load the data
df = pd.read_csv("GOOG.csv")

# Convert the 'date' column to datetime and then format it to just 'YYYY-MM-DD'
df['date'] = pd.to_datetime(df['date']).dt.date

# Drop unwanted columns
df = df.drop(columns=['symbol', 'adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume', 'divCash', 'splitFactor'], axis=1)

# Check for duplicates
duplicates_found = df.duplicated().values.any()

# Display data info
df.info()

# Descriptive statistics of the dataset
df_description = df.describe()

# Select only numeric columns for correlation analysis
numeric_df = df.select_dtypes(include=['number'])
correlations = numeric_df.corr()

# Heatmap of correlations
plt.figure(figsize=(16,8))
sns.heatmap(numeric_df.corr(), cmap="Blues", annot=True)
plt.show()

# Pairplot to check relationships between variables
sns.pairplot(df)

# Binning data (histograms for each feature)
columns_to_plot = ['open', 'high', 'low', 'close', 'volume']
for column in columns_to_plot:
    plt.figure(figsize=(10, 6))
    df[column].hist(bins=30, color='blue', edgecolor='black')
    plt.title(f'Distribution of {column.capitalize()}', fontsize=16)
    plt.xlabel(column.capitalize(), fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(axis='y', alpha=0.75)
    plt.show()

# Detecting outliers using boxplots
f, axes = plt.subplots(1, 4, figsize=(20,5))
sns.boxplot(y='open', data=df, ax=axes[0])
sns.boxplot(y='high', data=df, ax=axes[1])
sns.boxplot(y='low', data=df, ax=axes[2])
sns.boxplot(y='close', data=df, ax=axes[3])
plt.tight_layout()
plt.show()

# Detecting outliers with Z-score and scaling
z_scores = stats.zscore(df.select_dtypes(include=['number']))  # Excluding non-numeric columns
df_scaled = df.copy()
df_scaled[numeric_df.columns] = z_scores
df_no_outliers = df_scaled[(z_scores < 3).all(axis=1) & (z_scores > -3).all(axis=1)]
df_no_outliers.info()






figure = go.Figure(data=[go.Candlestick(x=df["date"],
                                        open=df["open"], high=df["high"],
                                        low=df["low"], close=df["close"])])
figure.update_layout(title = "Google Stock Price Analysis", xaxis_rangeslider_visible=False)
figure.show()
