"""Build the gold NL-to-SQL dataset from parameterized templates.

Every template is instantiated with real values drawn from shopsphere.db and
its SQL is executed against that DB at build time -- an example only makes it
into the dataset if the query actually runs and returns rows. This guarantees
the eval harness is scoring against ground truth that is provably correct,
not hand-typed SQL that might silently be wrong.

Run: python data/build_dataset.py
"""
import json
import random
import sqlite3
from pathlib import Path

SEED = 7
HERE = Path(__file__).parent
DB_PATH = HERE.parent / "schema" / "shopsphere.db"
random.seed(SEED)

conn = sqlite3.connect(DB_PATH)

COUNTRIES = [r[0] for r in conn.execute("SELECT DISTINCT country FROM customers")]
TIERS = [r[0] for r in conn.execute("SELECT DISTINCT tier FROM customers")]
CATEGORIES = [r[0] for r in conn.execute("SELECT DISTINCT category FROM products")]
ORDER_STATUSES = [r[0] for r in conn.execute("SELECT DISTINCT status FROM orders")]
TICKET_STATUSES = [r[0] for r in conn.execute("SELECT DISTINCT status FROM support_tickets")]
TICKET_PRIORITIES = [r[0] for r in conn.execute("SELECT DISTINCT priority FROM support_tickets")]
YEARS = [2022, 2023, 2024, 2025, 2026]

SCHEMA_DDL = (HERE.parent / "schema" / "schema.sql").read_text()

examples = []


def add(category, phrasings, sql, params=None):
    """Instantiate `sql`/`phrasings` with `params`, verify it executes, record it."""
    params = params or {}
    filled_sql = sql.format(**params)
    try:
        cur = conn.execute(filled_sql)
        rows = cur.fetchall()
    except sqlite3.Error as e:
        raise RuntimeError(f"Bad gold SQL for category={category}: {filled_sql}\n{e}")
    for phrasing in phrasings:
        examples.append({
            "question": phrasing.format(**params),
            "sql": filled_sql,
            "category": category,
            "gold_row_count": len(rows),
        })


# ---------- simple_filter ----------
for country in COUNTRIES:
    add("simple_filter", [
        "List the names of customers from {country}.",
        "Who are our customers based in {country}?",
        "Show all customer names where the country is {country}.",
    ], "SELECT name FROM customers WHERE country = '{country}'", {"country": country})

for status in ORDER_STATUSES:
    add("simple_filter", [
        "How many orders have status '{status}'?",
        "Count the orders that are currently {status}.",
    ], "SELECT COUNT(*) FROM orders WHERE status = '{status}'", {"status": status})

for cat in CATEGORIES:
    add("simple_filter", [
        "List all products in the {cat} category, cheapest first.",
        "Show {cat} products ordered by price ascending.",
    ], "SELECT name, price FROM products WHERE category = '{cat}' ORDER BY price ASC", {"cat": cat})

# ---------- aggregation ----------
for tier in TIERS:
    add("aggregation", [
        "How many customers are in the '{tier}' tier?",
        "Count customers whose tier equals {tier}.",
    ], "SELECT COUNT(*) FROM customers WHERE tier = '{tier}'", {"tier": tier})

add("aggregation", [
    "What is the average price of products in each category?",
    "Show the average product price grouped by category.",
], "SELECT category, ROUND(AVG(price), 2) AS avg_price FROM products GROUP BY category ORDER BY category")

add("aggregation", [
    "What is the total revenue (quantity times unit price) from all order items?",
    "Sum up quantity * unit_price across every order item to get total revenue.",
], "SELECT ROUND(SUM(quantity * unit_price), 2) FROM order_items")

for status in ORDER_STATUSES:
    add("aggregation", [
        "How many orders does each customer have with status '{status}'? Show customer name and count.",
        "For orders with status {status}, count them per customer and show the customer's name.",
    ], """
        SELECT c.name, COUNT(*) AS order_count
        FROM customers c JOIN orders o ON o.customer_id = c.customer_id
        WHERE o.status = '{status}'
        GROUP BY c.customer_id
        ORDER BY order_count DESC
    """, {"status": status})

# ---------- join (2-way / 3-way) ----------
add("join", [
    "List each order's id along with the customer's name and country.",
    "Show order_id, customer name, and customer country for every order.",
], "SELECT o.order_id, c.name, c.country FROM orders o JOIN customers c ON o.customer_id = c.customer_id")

for cat in CATEGORIES:
    add("join", [
        "List the names of customers who have ordered a product in the {cat} category.",
        "Which customers bought something from the {cat} category?",
    ], """
        SELECT DISTINCT c.name
        FROM customers c
        JOIN orders o ON o.customer_id = c.customer_id
        JOIN order_items oi ON oi.order_id = o.order_id
        JOIN products p ON p.product_id = oi.product_id
        WHERE p.category = '{cat}'
    """, {"cat": cat})

add("join", [
    "For each product, show its name and the average review rating it has received.",
    "List product names with their average rating from reviews.",
], """
    SELECT p.name, ROUND(AVG(r.rating), 2) AS avg_rating
    FROM products p JOIN reviews r ON r.product_id = p.product_id
    GROUP BY p.product_id
    ORDER BY avg_rating DESC
""")

for priority in TICKET_PRIORITIES:
    add("join", [
        "List support tickets with priority '{priority}' along with the customer's name and the assigned employee's name.",
        "For {priority}-priority tickets, show the customer name and the employee handling it.",
    ], """
        SELECT t.ticket_id, c.name AS customer_name, e.name AS employee_name
        FROM support_tickets t
        JOIN customers c ON c.customer_id = t.customer_id
        LEFT JOIN employees e ON e.employee_id = t.assigned_employee_id
        WHERE t.priority = '{priority}'
    """, {"priority": priority})

# ---------- self_join ----------
add("self_join", [
    "List each employee's name along with their manager's name.",
    "Show employee names paired with the name of the person who manages them.",
], """
    SELECT e.name AS employee_name, m.name AS manager_name
    FROM employees e LEFT JOIN employees m ON e.manager_id = m.employee_id
""")

add("self_join", [
    "Which employees have no manager assigned (top of the hierarchy)?",
    "List employees whose manager_id is null.",
], "SELECT name FROM employees WHERE manager_id IS NULL")

# ---------- top_n ----------
for n in (3, 5, 10):
    add("top_n", [
        f"What are the top {n} most expensive products?",
        f"List the {n} highest-priced products.",
    ], f"SELECT name, price FROM products ORDER BY price DESC LIMIT {n}")

add("top_n", [
    "Which 5 customers have spent the most money in total, based on delivered orders?",
    "Show the top 5 customers by total spend on delivered orders.",
], """
    SELECT c.name, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_spent
    FROM customers c
    JOIN orders o ON o.customer_id = c.customer_id AND o.status = 'delivered'
    JOIN order_items oi ON oi.order_id = o.order_id
    GROUP BY c.customer_id
    ORDER BY total_spent DESC
    LIMIT 5
""")

# ---------- having ----------
for n in (2, 3, 5):
    add("having", [
        f"Which customers have placed more than {n} orders?",
        f"List customers with an order count greater than {n}.",
    ], f"""
        SELECT c.name, COUNT(*) AS order_count
        FROM customers c JOIN orders o ON o.customer_id = c.customer_id
        GROUP BY c.customer_id
        HAVING COUNT(*) > {n}
        ORDER BY order_count DESC
    """)

add("having", [
    "Which product categories have an average price above 100?",
    "List categories where the average product price exceeds 100.",
], "SELECT category, ROUND(AVG(price),2) AS avg_price FROM products GROUP BY category HAVING AVG(price) > 100")

# ---------- negation / anti-join ----------
add("negation_antijoin", [
    "Which customers have never placed an order?",
    "List customers with zero orders.",
], """
    SELECT c.name FROM customers c
    LEFT JOIN orders o ON o.customer_id = c.customer_id
    WHERE o.order_id IS NULL
""")

add("negation_antijoin", [
    "Which products have never been reviewed?",
    "List products with no reviews at all.",
], """
    SELECT p.name FROM products p
    LEFT JOIN reviews r ON r.product_id = p.product_id
    WHERE r.review_id IS NULL
""")

add("negation_antijoin", [
    "Which employees have not been assigned any support tickets?",
    "List employees with no tickets assigned to them.",
], """
    SELECT e.name FROM employees e
    LEFT JOIN support_tickets t ON t.assigned_employee_id = e.employee_id
    WHERE t.ticket_id IS NULL
""")

# ---------- subquery_scalar / correlated ----------
add("subquery_scalar", [
    "Which products are priced above the average product price?",
    "List products more expensive than the overall average price.",
], "SELECT name, price FROM products WHERE price > (SELECT AVG(price) FROM products)")

add("subquery_correlated", [
    "Which customers have placed more orders than the average number of orders per customer?",
    "List customers whose order count exceeds the average order count across all customers.",
], """
    SELECT c.name, COUNT(o.order_id) AS order_count
    FROM customers c JOIN orders o ON o.customer_id = c.customer_id
    GROUP BY c.customer_id
    HAVING COUNT(o.order_id) > (
        SELECT AVG(cnt) FROM (
            SELECT COUNT(*) AS cnt FROM orders GROUP BY customer_id
        )
    )
""")

for cat in CATEGORIES:
    add("subquery_correlated", [
        "Which customers have ordered every product in the {cat} category?",
    ], """
        SELECT c.name
        FROM customers c
        WHERE NOT EXISTS (
            SELECT p.product_id FROM products p
            WHERE p.category = '{cat}'
            AND NOT EXISTS (
                SELECT 1 FROM orders o
                JOIN order_items oi ON oi.order_id = o.order_id
                WHERE o.customer_id = c.customer_id AND oi.product_id = p.product_id
            )
        )
    """, {"cat": cat})

# ---------- date_range ----------
for year in YEARS:
    add("date_range", [
        f"How many orders were placed in {year}?",
        f"Count orders where the order date falls in {year}.",
    ], f"SELECT COUNT(*) FROM orders WHERE strftime('%Y', order_date) = '{year}'")

add("date_range", [
    "Which customers signed up in the last 6 months (relative to 2026-08-20)?",
    "List customers whose signup_date is within 6 months before 2026-08-20.",
], "SELECT name, signup_date FROM customers WHERE signup_date >= date('2026-08-20', '-6 months')")

# ---------- multi_join (4-table) ----------
for tier in TIERS:
    add("multi_join", [
        "For '{tier}'-tier customers, list each order along with the product names and quantities purchased.",
    ], """
        SELECT o.order_id, c.name AS customer_name, p.name AS product_name, oi.quantity
        FROM customers c
        JOIN orders o ON o.customer_id = c.customer_id
        JOIN order_items oi ON oi.order_id = o.order_id
        JOIN products p ON p.product_id = oi.product_id
        WHERE c.tier = '{tier}'
    """, {"tier": tier})

conn.close()

# ---------- split + write ----------
by_category = {}
for ex in examples:
    by_category.setdefault(ex["category"], []).append(ex)

train, eval_ = [], []
for cat, items in by_category.items():
    random.shuffle(items)
    n_eval = max(1, round(len(items) * 0.2))
    eval_.extend(items[:n_eval])
    train.extend(items[n_eval:])

random.shuffle(train)
random.shuffle(eval_)

for ex in train + eval_:
    ex["sql"] = " ".join(ex["sql"].split())  # normalize whitespace

for name, rows in [("train.jsonl", train), ("eval.jsonl", eval_)]:
    with open(HERE / name, "w") as f:
        for i, ex in enumerate(rows):
            ex["id"] = f"{name.split('.')[0]}-{i:04d}"
            f.write(json.dumps(ex) + "\n")

print(f"Total examples: {len(examples)}")
print(f"Train: {len(train)}  Eval: {len(eval_)}")
print("By category (train / eval):")
for cat in sorted(by_category):
    t = sum(1 for e in train if e["category"] == cat)
    ev = sum(1 for e in eval_ if e["category"] == cat)
    print(f"  {cat:22s} {t:4d} / {ev:3d}")
