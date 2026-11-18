import pandas as pd
import pandas_ta as ta

# Load the CSV file
file_path = r"D:/ohlc/1min/data_ohlc_feb13_1mnts.csv"
df = pd.read_csv(file_path)

print (df,'\n\n')

# Ensure 'close' column exists
if 'close' not in df.columns:
    raise ValueError("Column 'close' not found in the dataset.")

# Calculate 34 EMA using pandas_ta
df['34ema'] = ta.ema(df['close'], length=34)

# Initialize the result list
ema_differences = []

# Perform the subtraction operation (ema_34[j] - ema_34[j-4])
for j in range(4, len(df)):  # Start from index 4 to ensure j-4 is valid
    ema_diff = df.loc[j, '34ema'] - df.loc[j-4, '34ema']
    ema_differences.append(ema_diff)

# Print the final list
print(sorted(ema_differences, reverse=True))