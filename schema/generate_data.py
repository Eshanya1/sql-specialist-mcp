"""Populate shopsphere.db from schema.sql with seeded synthetic data.

Deterministic (seed=42) so the eval dataset's gold SQL results never drift.
Run: python schema/generate_data.py
"""
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

SEED = 42
HERE = Path(__file__).parent
DB_PATH = HERE / "shopsphere.db"
SCHEMA_PATH = HERE / "schema.sql"

random.seed(SEED)

COUNTRIES = ["USA", "UK", "Germany", "India", "Canada", "Australia", "Brazil", "Japan"]
TIERS = ["free", "silver", "gold", "platinum"]
TIER_WEIGHTS = [0.5, 0.25, 0.18, 0.07]
CATEGORIES = {
    "Electronics": ["Wireless Mouse", "Mechanical Keyboard", "USB-C Hub", "Noise-Cancelling Headphones",
                    "27-inch Monitor", "Webcam", "Portable SSD", "Bluetooth Speaker"],
    "Home": ["Desk Lamp", "Standing Desk", "Ergonomic Chair", "Air Purifier", "Coffee Maker",
             "Throw Blanket", "Cast Iron Pan", "Digital Kitchen Scale"],
    "Books": ["Deep Learning textbook", "Sci-Fi Anthology", "Cookbook: Weeknight Dinners",
              "History of Computing", "Mystery Novel", "Poetry Collection"],
    "Sports": ["Yoga Mat", "Resistance Bands", "Running Shoes", "Water Bottle", "Foam Roller"],
    "Toys": ["Building Blocks Set", "Puzzle 1000pc", "RC Car", "Board Game: Strategy"],
}
FIRST_NAMES = ["Aria", "Liam", "Maya", "Noah", "Zara", "Kai", "Priya", "Ethan", "Sofia", "Omar",
               "Chloe", "Ravi", "Emma", "Jin", "Layla", "Owen", "Nadia", "Lucas", "Ines", "Theo"]
LAST_NAMES = ["Chen", "Patel", "Garcia", "Smith", "Kim", "Mueller", "Silva", "Nakamura", "Khan", "Rossi"]
ROLES = ["Support Agent", "Senior Support Agent", "Support Lead", "Support Manager"]
TICKET_SUBJECTS = ["Order not delivered", "Wrong item received", "Refund request",
                    "Product defective", "Billing question", "Account access issue",
                    "Shipping address change", "Cancel order request"]


def rand_date(start: date, end: date) -> str:
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, max(delta, 0)))).isoformat()


def build():
    DB_PATH.unlink(missing_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())

    today = date(2026, 8, 20)
    epoch = date(2022, 1, 1)

    # employees (with a small management hierarchy for self-join queries)
    employees = []
    for i in range(1, 13):
        role = ROLES[0] if i > 4 else ROLES[min(i - 1, 3)]
        manager_id = None if i <= 2 else random.choice([1, 2])
        employees.append((i, f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}", role,
                           rand_date(epoch, today - timedelta(days=30)), manager_id))
    conn.executemany("INSERT INTO employees VALUES (?,?,?,?,?)", employees)

    # customers
    n_customers = 400
    customers = []
    for i in range(1, n_customers + 1):
        fn, ln = random.choice(FIRST_NAMES), random.choice(LAST_NAMES)
        signup = rand_date(epoch, today)
        customers.append((i, f"{fn} {ln}", f"{fn.lower()}.{ln.lower()}{i}@example.com",
                           random.choice(COUNTRIES), signup,
                           random.choices(TIERS, weights=TIER_WEIGHTS)[0]))
    conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?)", customers)

    # products
    products = []
    pid = 1
    for category, names in CATEGORIES.items():
        for name in names:
            price = round(random.uniform(8, 450), 2)
            products.append((pid, name, category, price, random.randint(0, 500)))
            pid += 1
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?)", products)
    n_products = pid - 1

    # orders + order_items
    orders, order_items = [], []
    oid, oiid = 1, 1
    statuses = ["pending", "shipped", "delivered", "cancelled", "returned"]
    status_weights = [0.08, 0.12, 0.65, 0.08, 0.07]
    for cust in customers:
        cust_id, signup_str = cust[0], cust[4]
        signup = date.fromisoformat(signup_str)
        n_orders = random.choices([0, 1, 2, 3, 4, 5, 8], weights=[0.15, 0.2, 0.2, 0.15, 0.12, 0.1, 0.08])[0]
        for _ in range(n_orders):
            odate = rand_date(signup, today)
            status = random.choices(statuses, weights=status_weights)[0]
            orders.append((oid, cust_id, odate, status))
            n_items = random.randint(1, 4)
            for _ in range(n_items):
                prod = products[random.randint(0, n_products - 1)]
                qty = random.randint(1, 3)
                order_items.append((oiid, oid, prod[0], qty, prod[3]))
                oiid += 1
            oid += 1
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?)", orders)
    conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", order_items)

    # reviews (only from customers who ordered, roughly)
    reviews = []
    rid = 1
    delivered_orders = [o for o in orders if o[3] in ("delivered", "returned")]
    for o in delivered_orders:
        if random.random() < 0.35:
            items = [it for it in order_items if it[1] == o[0]]
            if not items:
                continue
            item = random.choice(items)
            rating = random.choices([1, 2, 3, 4, 5], weights=[0.05, 0.07, 0.15, 0.33, 0.4])[0]
            reviews.append((rid, item[2], o[1], rating, rand_date(date.fromisoformat(o[2]), today), None))
            rid += 1
    conn.executemany("INSERT INTO reviews VALUES (?,?,?,?,?,?)", reviews)

    # support tickets
    tickets = []
    tid = 1
    agent_ids = [e[0] for e in employees]
    for cust in customers:
        if random.random() < 0.25:
            cust_orders = [o for o in orders if o[1] == cust[0]]
            order_ref = random.choice(cust_orders)[0] if cust_orders and random.random() < 0.7 else None
            created = rand_date(date.fromisoformat(cust[4]), today)
            tickets.append((tid, cust[0], order_ref, random.choice(agent_ids),
                             random.choices(["open", "in_progress", "resolved", "closed"],
                                             weights=[0.15, 0.15, 0.2, 0.5])[0],
                             random.choices(["low", "medium", "high", "urgent"],
                                             weights=[0.4, 0.35, 0.18, 0.07])[0],
                             created))
            tid += 1
    conn.executemany("INSERT INTO support_tickets VALUES (?,?,?,?,?,?,?)", tickets)

    conn.commit()
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ["customers", "employees", "products", "orders", "order_items", "reviews", "support_tickets"]}
    conn.close()
    print(f"Built {DB_PATH} (seed={SEED})")
    for t, c in counts.items():
        print(f"  {t}: {c} rows")


if __name__ == "__main__":
    build()
