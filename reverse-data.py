import pandas as pd

# Read the CSV file
df = pd.read_csv('fara_bourse_2009-09-28_2024-10-23.csv', parse_dates=['Price'])

# Sort the dataframe by date, oldest first
df_sorted = df.sort_values('Price', ascending=True)

# Reset the index
df_sorted = df_sorted.reset_index(drop=True)

# Save the sorted dataframe to a new CSV file
df_sorted.to_csv('reveresed_fara_bourse_2009-09-28_2024-10-23.csv', index=False)

print("File has been sorted and saved as 'fara_bourse_2009-09-28_2024-10-23.csv'")

