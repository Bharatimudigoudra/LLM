import pandas as pd
from langchain_groq import ChatGroq

def analyze_dataset(file_path):
    # Load the dataset
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
    
    # Prepare dataset summary
    summary = []
    summary.append(f"Number of Rows: {df.shape[0]}")
    summary.append(f"Number of Columns: {df.shape[1]}")
    summary.append("\nData Types of Each Column:")
    summary.append(df.dtypes.to_string())
    summary.append("\nSummary Statistics of Numerical Columns:")
    summary.append(df.describe().to_string())
    summary.append("\nMissing Values in Each Column:")
    summary.append(df.isnull().sum().to_string())
    summary.append("\nUnique Values in Each Column:")
    for col in df.columns:
        summary.append(f"{col}: {df[col].nunique()} unique values")
    summary.append("\nDetailed Analysis of Each Feature:")
    for col in df.columns:
        summary.append(f"\nFeature: {col}")
        if df[col].dtype == 'object':
            summary.append("Feature Type: Categorical")
            summary.append(f"Number of Unique Categories: {df[col].nunique()}")
            summary.append(f"Most Frequent Category: {df[col].mode()[0]}")
            summary.append(f"Frequency of Most Frequent Category: {df[col].value_counts().iloc[0]}")
            summary.append(f"Top Categories: \n{df[col].value_counts().head()}")
        else:
            summary.append("Feature Type: Numerical")
            summary.append(f"Mean: {df[col].mean()}")
            summary.append(f"Median: {df[col].median()}")
            summary.append(f"Standard Deviation: {df[col].std()}")
            summary.append(f"Minimum Value: {df[col].min()}")
            summary.append(f"Maximum Value: {df[col].max()}")
            summary.append(f"25th Percentile: {df[col].quantile(0.25)}")
            summary.append(f"50th Percentile (Median): {df[col].quantile(0.5)}")
            summary.append(f"75th Percentile: {df[col].quantile(0.75)}")
        
        if df[col].dtype == 'object':
            unique_types = df[col].apply(type).nunique()
            if unique_types > 1:
                summary.append("Note: This column contains mixed data types.")
    
    summary_text = "\n".join(summary)
    
    # Set up the Groq model with API key
    api_key = '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'  # Replace with your actual API key
    model = ChatGroq(model_name="llama-3.1-70b-versatile", api_key=api_key)  # Replace with your Groq model name
    
    # Get the model response by passing the summary text directly
    try:
        response = model.invoke(summary_text)
        print("Model Response:")
        print(response.content)  # Access the content of the AIMessage directly
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage:
file_path = r'C:\Bharati\ineuron_project_2\uploads\zomato.csv'  # Replace with your file path
analyze_dataset(file_path)
