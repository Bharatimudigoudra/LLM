import pandas as pd
import json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Initialize ChatGroq model
model = ChatGroq(model_name="llama-3.1-70b-versatile", api_key = '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')  # Replace with actual model name

def generate_insights(df):
    """
    Generate insights for each column of the dataset using ChatGroq.
    """
    # Convert dataset sample to JSON format for the prompt
    sample_data = df.head().to_dict(orient='records')
    sample_data_json = json.dumps(sample_data, indent=2)

    # Create a prompt for the LLM
    prompt_text = (
        f"Analyze the following dataset and provide insights for each column. Describe the contents, "
        f"what each column tells about the data, and what kind of features can be extracted from each column. "
        f"Dataset:\n\n{sample_data_json}"
    )
    
    # Prepare the message for the model
    messages = [HumanMessage(content=prompt_text)]

    try:
        # Invoke the model with the message
        response = model.invoke(messages)
        
        # Extract and print the content from the response
        if response and hasattr(response, 'content'):
            print("\n**Model Response:**")
            print(response.content)
        else:
            print("\nAn error occurred: Response content is not accessible.")
    
    except Exception as e:
        print(f"\nAn error occurred: {e}")

def main(file_path):
    """
    Main function to read the dataset and generate insights.
    """
    # Read dataset
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')

    # Print basic information about the dataset
    print("Dataset Overview:")
    print(df.head())
    print(df.info())
    
    # Generate insights
    generate_insights(df)

if __name__ == "__main__":
    # Replace 'your_dataset.csv' with your dataset file path
    main(r'C:\Bharati\ineuron_project_2\uploads\used_cars.csv')
