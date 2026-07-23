import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_demo_dataset(months: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates a realistic historical dataset of orders and order lines 
    spanning the specified number of months for demo and testing purposes.
    """
    np.random.seed(42)
    end_date = pd.Timestamp.now().floor("D")
    start_date = end_date - pd.DateOffset(months=months)
    
    customers = [
        "Acme Corporation", "Global Logistics LLC", "Nexus Tech Solutions",
        "Starlight Retail", "Apex Manufacturing", "Zenith Health",
        "Vanguard Systems", "Horizon Energy", "Summit Financial", "Pinnacle Media"
    ]
    
    products = [
        {"name": "Enterprise Server X1", "category": "Hardware", "price": 2500.0},
        {"name": "Cloud Storage Node", "category": "Hardware", "price": 1200.0},
        {"name": "Pro Workstation Unit", "category": "Hardware", "price": 1800.0},
        {"name": "SaaS Platform License", "category": "Software", "price": 450.0},
        {"name": "Security Suite Pro", "category": "Software", "price": 890.0},
        {"name": "Database Analytics Module", "category": "Software", "price": 1100.0},
        {"name": "24/7 Premium Support", "category": "Services", "price": 300.0},
        {"name": "Implementation Consulting", "category": "Services", "price": 1500.0},
    ]

    orders = []
    order_lines = []

    order_counter = 1000
    line_counter = 5000

    # Generate dates across history
    current_date = start_date
    while current_date <= end_date:
        # Seasonality: more orders towards Q4
        seasonality = 1.0 + 0.3 * np.sin(2 * np.pi * current_date.month / 12)
        # Trend: steady 1% monthly growth
        months_passed = (current_date.year - start_date.year) * 12 + (current_date.month - start_date.month)
        trend = 1.0 + (0.015 * months_passed)
        
        num_orders = int(np.random.poisson(lam=8 * seasonality * trend))
        
        for _ in range(max(1, num_orders)):
            order_counter += 1
            order_ref = f"SO{order_counter}"
            customer = np.random.choice(customers)
            order_date = current_date + pd.Timedelta(days=int(np.random.randint(0, 28)))
            if order_date > end_date:
                continue

            num_items = np.random.randint(1, 4)
            order_subtotal = 0.0
            
            for _ in range(num_items):
                line_counter += 1
                prod = np.random.choice(products)
                qty = int(np.random.randint(1, 10))
                price = prod["price"]
                disc = float(np.random.choice([0, 5, 10], p=[0.7, 0.2, 0.1]))
                line_total = qty * price * (1.0 - disc / 100.0)
                order_subtotal += line_total
                
                order_lines.append({
                    "order_lines": f"LINE{line_counter}",
                    "order_reference": order_ref,
                    "quantity": qty,
                    "product": prod["name"],
                    "product_name": prod["name"],
                    "product_category": prod["category"],
                    "delivery_quantity": qty,
                    "unit_price": price,
                    "subtotal": round(line_total, 2),
                    "discount_(%)": disc
                })

            tax = round(order_subtotal * 0.1, 2)
            total = round(order_subtotal + tax, 2)

            orders.append({
                "order_date": order_date.strftime("%Y-%m-%d"),
                "order_reference": order_ref,
                "customer": customer,
                "total": total,
                "untaxed_amount": round(order_subtotal, 2),
                "status": "Sales Order"
            })

        current_date += pd.DateOffset(months=1)

    orders_df = pd.DataFrame(orders)
    order_lines_df = pd.DataFrame(order_lines)

    return orders_df, order_lines_df

if __name__ == "__main__":
    orders_df, order_lines_df = generate_demo_dataset()
    print(f"Generated {len(orders_df)} orders and {len(order_lines_df)} order lines.")
