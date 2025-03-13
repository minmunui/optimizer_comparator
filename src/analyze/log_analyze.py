import os
import pandas as pd

path = "logs/da_iter_frac4"
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
    params = file.split("!")[1:]

    record = {}
    for param in params:
        key, value = param.split("-")
        record[key] = value

    with open(f"{path}/{file}", 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "Fitness: " in line:
                value = trim_to_number(line.split(":")[1])
                record["value"] = value
            if "Total elapsed time: " in line:
                time = trim_to_number(line.split(":")[1])
                record["time"] = time

    results.append(record)

result_df = pd.concat([pd.DataFrame([result]) for result in results], ignore_index=True)
result_df.to_csv("summary.csv", index=False)
