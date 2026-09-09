import json, sys, math

numbers = [float(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [10.0, 20.0, 30.0, 40.0, 50.0]
count = len(numbers)
mean_val = sum(numbers) / count
sorted_nums = sorted(numbers)
mid = count // 2
median_val = (sorted_nums[mid] if count % 2 == 1 else (sorted_nums[mid - 1] + sorted_nums[mid]) / 2.0)
variance = sum((x - mean_val) ** 2 for x in numbers) / count
std_dev = math.sqrt(variance)

result = {
    "count": count,
    "numbers": numbers,
    "mean": round(mean_val, 2),
    "median": round(median_val, 2),
    "std_dev": round(std_dev, 3)
}
print(json.dumps(result))
