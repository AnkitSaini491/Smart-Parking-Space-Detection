

import pandas as pd
import matplotlib.pyplot as plt

# Sample traffic data
data = {
    "Vehicle Type": ["Car", "Motorcycle", "Bus", "Truck"],
    "Count": [150, 90, 35, 25]
}

df = pd.DataFrame(data)

# Total vehicles
total_vehicles = df["Count"].sum()

print("AI SMART TRAFFIC ANALYSIS")
print("-------------------------")

print("Total Vehicles:", total_vehicles)

# Vehicle statistics
print("\nVehicle Count:")
print(df)

# Find highest vehicle category
highest_vehicle = df.loc[
    df["Count"].idxmax(), "Vehicle Type"
]

print("\nMost Common Vehicle:", highest_vehicle)

# Calculate percentage
df["Percentage"] = (
    df["Count"] / total_vehicles
) * 100

print("\nVehicle Percentage:")
print(df)

# Bar Chart
plt.figure(figsize=(8, 5))

plt.bar(df["Vehicle Type"], df["Count"])

plt.title("Vehicle Count Analysis")
plt.xlabel("Vehicle Type")
plt.ylabel("Number of Vehicles")

plt.tight_layout()
plt.savefig("vehicle_analysis.png")
plt.show()

# Pie Chart
plt.figure(figsize=(7, 7))

plt.pie(
    df["Count"],
    labels=df["Vehicle Type"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Traffic Distribution")

plt.savefig("traffic_distribution.png")
plt.show()

# Save analysis report
df.to_csv("traffic_analysis_report.csv", index=False)

print("\nAnalysis report saved successfully!")
