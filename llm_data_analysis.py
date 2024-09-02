import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage #LangChain often requires input to be formatted as a list of message objects (e.g., HumanMessage, SystemMessage, AIMessage).

# Set the Groq API key (using environment variable for security)
# use command prompt "set GROQ_API_KEY=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"
os.environ["GROQ_API_KEY"] = "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"

# Initialize the LLM (Groq model via LangChain)
llm = ChatGroq()

# Load the dataset
df = pd.read_csv(r'C:\Bharati\ineuron_project_2\uploads\streeteasy.csv')

# Display basic information to understand the data types
print("Original DataFrame:")
print(df.head())
print(df.info())

# Handle missing values (if any) - Dropping for simplicity
df = df.dropna()

# Automatically detect numerical and categorical features
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object', 'category']).columns


# Analyze the dataset and request plot suggestions from the LLM
analysis_prompt = f"""
I have a dataset with the following features:
Numeric Features: {', '.join(numeric_features)}
Categorical Features: {', '.join(categorical_features)}

Please suggest the most appropriate plots to create for each feature and between pairs of features.
"""

# Use HumanMessage to wrap the input
message = HumanMessage(content=analysis_prompt)

# Use the invoke method
plot_suggestions = llm.invoke([message])
print("\nLLM Plot Suggestions:")
print(plot_suggestions)

# Example: Based on the LLM's suggestions, generate plots (you'll adjust these based on actual suggestions)
# Generate plots for numeric features
for feature in numeric_features:
    plt.figure(figsize=(10, 6))
    
    # Distribution plot
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

    # Boxplot to visualize distribution and outliers
    sns.boxplot(x=df[feature])
    plt.title(f'Boxplot of {feature}')
    plt.show()

# Generate plots for categorical features
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    
    # Count plot to visualize the frequency of categories
    sns.countplot(x=df[feature])
    plt.title(f'Count Plot of {feature}')
    plt.xticks(rotation=45)
    plt.show()

    # If the dataset has numerical columns, create a bar plot for the relationship between categorical and numerical features
    for num_feature in numeric_features:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=df[feature], y=df[num_feature])
        plt.title(f'{num_feature} vs {feature}')
        plt.xticks(rotation=45)
        plt.show()

# Generate pairplots and heatmaps if suggested by the LLM
if len(numeric_features) > 1:
    sns.pairplot(df[numeric_features])
    plt.title('Pairplot of Numeric Features')
    plt.show()

    correlation_matrix = df[numeric_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.show()
