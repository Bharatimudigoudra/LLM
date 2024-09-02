import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r'C:\Bharati\ineuron_project_2\uploads\Employee.csv')

# Display basic information to understand the data types
print("Original DataFrame:")
print(df.head())
print(df.info())

# Handle missing values (if any) - Dropping for simplicity
df = df.dropna()

# Function to determine if a column is mixed with int and float
def is_mixed_numeric(series):
    return series.apply(lambda x: isinstance(x, (int, float))).all() and series.apply(lambda x: isinstance(x, float)).any()

# Process each column
for col in df.columns:
    if df[col].dtype == 'int64' or df[col].dtype == 'float64':
        # Leave pure integer columns unchanged
        continue
    elif is_mixed_numeric(df[col]):
        # Convert mixed numeric columns to float
        df[col] = df[col].astype(float)
    elif df[col].dtype == 'object':
        # Convert object/string columns to categorical if they represent categories
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes
    else:
        # Convert any other columns to float as a fallback
        df[col] = df[col].astype(float)

df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
df.columns = df.columns.str.lower()  # Convert to lower case

# Display the updated DataFrame to verify changes
print("\nUpdated DataFrame:")
print(df.head())
print(df.info())
