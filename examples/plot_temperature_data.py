import pandas as pd
import matplotlib.pyplot as plt
import os

# Load temperature data
csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'example_temperature_data.csv')
df = pd.read_csv(csv_path)
df['date'] = pd.to_datetime(df['date'])

plt.figure(figsize=(10, 5))
plt.plot(df['date'], df['temperature'], marker='o', color='steelblue', linewidth=2)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Annual Mean Temperature (°C)', fontsize=12)
plt.title('Annual Mean Temperature in Norway (1960-2025)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
