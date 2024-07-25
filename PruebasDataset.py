import pandas as pd

# Step 1: Load the CSV file with utf-8 encoding and flexible delimiter
file_path = 'newDataV2.csv'
df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', sep=None, engine='python')

# Step 2: Check for missing values and fill or drop them if necessary
df.fillna('', inplace=True)  # Fill missing values with empty strings for simplicity

# Step 3: Define a function to create labeled text
def create_labeled_text(row):
    return f"Company: {row['empresa']} Segment: {row['segmento']} Emotion: {row['emocion']} Description: {row['descripcion']} Slogan: {row['eslogan']}"

# Step 4: Apply the function to create a new column with labeled text
df['labeled_text'] = df.apply(create_labeled_text, axis=1)

# Step 5: Save the labeled text to a .txt file
txt_file_path = 'EntrenoNum4.txt'
df['labeled_text'].to_csv(txt_file_path, index=False, header=False)

# Step 6: Display the saved file path
print(txt_file_path)

