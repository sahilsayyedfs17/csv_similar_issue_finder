import pandas as pd
import re

# Function to remove floating-point values from a text
def remove_floats(text):
    # Replace floating-point numbers with an empty string
    return re.sub(r'\b\d+\.\d+\b', '', text)

# Load the CSV file into a DataFrame
csv_file_path = 'vtvas_issues_clean.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file_path)

# Process the Description and Summary fields
df['Description'] = df['Description'].astype(str).apply(remove_floats)
df['Summary'] = df['Summary'].astype(str).apply(remove_floats)

# Save the cleaned DataFrame to a new CSV file
output_csv_file_path = 'cleaned_VTVAS.csv'  # Replace with your desired output path
df.to_csv(output_csv_file_path, index=False)

print(f"Processed CSV file saved to: {output_csv_file_path}")
