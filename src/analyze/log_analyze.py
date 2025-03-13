import os
import pandas as pd

path = "logs/ga_iter_frac4"
files = os.listdir(path)

def trim_to_number(s: str) -> float:
    start = 0
    end = len(s)
    for i in range(len(s)):
        if is_number(s[i]):
            start = i
            break
    for i in range(len(s) - 1, -1, -1):
        if is_number(s[i]):
            end = i + 1
            break
    return float(s[start:end])

def is_number(char):
    if char in "0123456789.":
        return True
    return False

results = []

for file in files:
    mutation_rate, crossover_rate, seed = file.split("_")[1:4]
    mutation_rate = trim_to_number(mutation_rate)
    crossover_rate = trim_to_number(crossover_rate)
    seed = trim_to_number(seed)

    with open(f"{path}/{file}", 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "Fitness: " in line:
                value = trim_to_number(line.split(":")[1])
            if "Total elapsed time: " in line:
                time = trim_to_number(line.split(":")[1])

    results.append({"mutation_rate": mutation_rate, "crossover_rate": crossover_rate, "seed": seed, "value": value, "time": time})

result_df = pd.concat([pd.DataFrame([result]) for result in results], ignore_index=True)
result_df.to_csv("summary.csv", index=False)