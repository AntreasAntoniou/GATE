# Define the times
times = {
    "Line 1": 3.337860107421875e-06,
    "Line 5": 2.2411346435546875e-05,
    "Line 7": 5.030632019042969e-05,
    "Line 4": 0.00012421607971191406,
    "Line 3": 0.000423431396484375,
    "Line 9": 0.0005199909210205078,
    "Line 14": 0.0005292892456054688,
    "Line 12": 0.0006825923919677734,
    "Line 11": 0.0006933212280273438,
    "Line 10": 0.0006990432739257812,
    "Line 15": 0.0008726119995117188,
    "Line 13": 0.0010602474212646484,
    "Line 8": 0.0011425018310546875,
    "Line 13": 0.0011394023895263672,
    "Line 6": 0.010357379913330078,
    "Line 2": 0.01126408576965332,
}

# Calculate the total time
total_time = sum(times.values())

# Calculate the percentage for each line and sort in descending order
percentages = {line: (time / total_time) * 100 for line, time in times.items()}
sorted_percentages = sorted(
    percentages.items(), key=lambda x: x[1], reverse=True
)

# Print the sorted list
for line, percentage in sorted_percentages:
    print(f"{line}: {times[line]} seconds ({percentage:.2f}%)")
