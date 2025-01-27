import time
import os
import pandas as pd
from smolagents import CodeAgent, LiteLLMModel, Tool, tool
from smolagents.default_tools import PythonInterpreterTool
from dotenv import load_dotenv

load_dotenv()

# Rate limiting parameters
RATE_LIMIT_REQUESTS = 25  # Allow 25 requests...
RATE_LIMIT_PERIOD = 60  # ...per 60 seconds

# Define the DataAnalysisTool
class DataAnalysisTool(Tool):
    name = "data_analyzer"
    description = "Loads a CSV file into a pandas DataFrame and provides basic analysis (e.g., mean, median, standard deviation) of a specified column. The filepath is relative to where you launched the script."
    inputs = {
        "filepath": {
            "type": "string",
            "description": "Path to the CSV file. This can be a file that you have uploaded to the session.",
        },
        "column": {
            "type": "string",
            "description": "The name of the column to analyze.",
        },
    }
    output_type = "string"

    def forward(self, filepath: str, column: str) -> str:
        try:
            df = pd.read_csv(filepath)
            if column not in df.columns:
                return f"Error: Column '{column}' not found in the DataFrame."

            mean = df[column].mean()
            median = df[column].median()
            std_dev = df[column].std()

            result = (
                f"Data Analysis for column '{column}' in file '{filepath}':\n"
                f"  Mean: {mean:.2f}\n"
                f"  Median: {median:.2f}\n"
                f"  Standard Deviation: {std_dev:.2f}"
            )
            return result
        except FileNotFoundError:
            return f"Error: File not found at path: {filepath}"
        except Exception as e:
            return f"Error during data analysis: {e}"

# Initialize the Groq model using LiteLLM
your_groq_model = LiteLLMModel(model_id="groq/llama-3.3-70b-versatile")

# Create tool instances
data_analysis_tool = DataAnalysisTool()
python_interpreter_tool = PythonInterpreterTool()

# Initialize the CodeAgent
code_agent = CodeAgent(
    model=your_groq_model,
    tools=[python_interpreter_tool, data_analysis_tool],
    additional_authorized_imports=["pandas", "tempfile", "os"],
    # use_e2b_executor=True,  # Removed E2B
)

# Example CSV data (replace with your actual data)
csv_data = """product,price,quantity
apple,1.0,10
banana,0.5,20
cherry,0.2,50
"""
with open("sales_data.csv", "w") as f:
    f.write(csv_data)

# Run the agent
try:
    result = code_agent.run(
        "Analyze the 'price' column in the dataset 'sales_data.csv' that I have uploaded to the session. Also, calculate the total value of all sales (price * quantity) and print it.",
        additional_args={"sales_data.csv": "sales_data.csv"}
    )
    print(result)
except Exception as e:
    print(f"An error occurred: {e}")