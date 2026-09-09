def calculate_discounted_price(price: float, discount_percent: float) -> float:
    if price < 0.0:
        raise ValueError("Price cannot be negative.")
    if discount_percent < 0.0 or discount_percent > 100.0:
        raise ValueError("Discount percentage must be between 0.0 and 100.0.")
    discount_amount = price * (discount_percent / 100.0)
    return round(price - discount_amount, 2)
