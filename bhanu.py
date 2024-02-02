import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
train_data = pd.read_excel("Dataset1.xlsx", sheet_name="train")
test_data = pd.read_excel("Dataset1.xlsx", sheet_name="test")
ideal_data = pd.read_excel("Dataset1.xlsx", sheet_name="ideal")

# Task-1: Evaluate the Least Square Error for Each Column
column_indices = []
least_square_errors = []

for train_col_index in range(1, len(train_data.columns)):
    lse_list = []
    for ideal_col_index in range(1, len(ideal_data.columns)):
        mse = 0
        for row_index in range(len(train_data)):
            train_value = train_data.iloc[row_index, train_col_index]
            ideal_value = ideal_data.iloc[row_index, ideal_col_index]
            mse += ((train_value - ideal_value) ** 2)
        lse_list.append(mse / len(train_data))
    min_lse = min(lse_list)
    index = lse_list.index(min_lse)
    column_indices.append(index + 1)
    least_square_errors.append(min_lse)

task1_results = pd.DataFrame({"Selected_Column_Index": column_indices, "Least_Square_Error": least_square_errors})

# Task-2: Integrate Ideal Functions into Test Data
merged_ideal_functions = []
for i in range(0, len(task1_results)):
    merged_ideal_functions.append(ideal_data[["x", f"y{str(task1_results.iloc[i, 0])}"]])

for ideal_function in merged_ideal_functions:
    test_data = test_data.merge(ideal_function, on='x', how='left')

# Calculate Deviation and Determine Ideal Index
ideal_indices = []
deviations = []

for row_index in range(len(test_data)):
    mse_list = []
    for test_col_index in range(2, len(test_data.columns)):
        v1 = test_data.iloc[row_index, 1]
        v2 = test_data.iloc[row_index, test_col_index]
        mse = ((v1 - v2) ** 2)
        mse_list.append(mse)

    min_mse = min(mse_list)
    if min_mse < np.sqrt(2) * 0.089005:
        deviations.append(min_mse)
        index = mse_list.index(min_mse)
        ideal_indices.append(index)
    else:
        deviations.append(min_mse)
        ideal_indices.append('miss')

test_data["Deviation"] = deviations
test_data["Ideal_Index"] = ideal_indices

# Data Visualization
 # Visualize Least Square Error for Each Selected Column (Task-1)

plt.figure(figsize=(10, 6))
sns.barplot(data=task1_results, x="Selected_Column_Index", y="Least_Square_Error")
plt.title("Task-1: Visualize Least Square Error for Each Selected Column")
plt.xlabel("Selected Column Index")
plt.ylabel("Least Square Error")
plt.xticks(rotation=45)
plt.show()

 # Visualize Merged Columns vs. x (Task-2)

plt.figure(figsize=(10, 6))
for i in range(0, len(task1_results)):
    plt.plot(test_data["x"], test_data[f"y{str(task1_results.iloc[i, 0])}"],
             label=f"y{str(task1_results.iloc[i, 0])} (Merged)", marker='o')

plt.title("Task-2: Visualize Merged Columns vs. x")
plt.xlabel("x")
plt.ylabel("Values")
plt.legend()
plt.show()

# Visualize Deviation Patterns Across x Values
plt.figure(figsize=(10, 6))
sns.scatterplot(data=test_data, x="x", y="Deviation")
plt.title("Visualize Deviation Patterns Across x Values")
plt.xlabel("x")
plt.ylabel("Deviation")
plt.show()

# Visualize Distribution of Ideal Indices

plt.figure(figsize=(8, 6))
sns.countplot(data=test_data, x="Ideal_Index")
plt.title(" Visualize Distribution of Ideal Indices")
plt.xlabel("Ideal Index")
plt.ylabel("Count")
plt.show()