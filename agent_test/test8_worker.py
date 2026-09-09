def compute_weighted_average(scores, weights):
    if not scores or not weights:
        return 0.0
    # Safe float coercion
    num_scores = [float(s) for s in scores]
    num_weights = [float(w) for w in weights]
    total_weight = sum(num_weights)
    if total_weight == 0.0:
        return 0.0
    total_val = sum(s * w for s, w in zip(num_scores, num_weights))
    return round(total_val / total_weight, 2)
