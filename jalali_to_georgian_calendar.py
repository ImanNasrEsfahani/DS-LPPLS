import pandas as pd
from persiantools.jdatetime import JalaliDate

def convert_jalali_to_gregorian(input_csv, output_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Check if the expected columns are present
    if 'dateissue' not in df.columns or 'Value' not in df.columns:
        raise ValueError("CSV must contain 'dateissue' and 'Value' columns.")


    # Convert Jalali dates to Gregorian and create a new DataFrame
    gregorian_dates = []
    for jalali_date in df['dateissue']:
        jalali_date_str = str(jalali_date)
        # Split the Jalali date into year, month, and day using slicing
        year = int(jalali_date_str[:4])   # First 4 digits for the year
        month = int(jalali_date_str[4:6])  # Next 2 digits for the month
        day = int(jalali_date_str[6:8])    # Last 2 digits for the day

        # Convert to Gregorian date
        gregorian_date = JalaliDate(year, month, day).to_gregorian()
        gregorian_dates.append(gregorian_date)

    # Create a new DataFrame with datetime and closes
    result_df = pd.DataFrame({
        'datetime': gregorian_dates,
        'close': df['Value']
    })

    # Save the new DataFrame to a CSV file
    result_df.to_csv(output_csv, index=False)
    print(f"Converted data saved to {output_csv}")

# Example usage
convert_jalali_to_gregorian(rf'data\Fara-bourse.csv', rf'data\Fara-bourse_georgian-calndar.csv')
