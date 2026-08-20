-- ShopSphere: a synthetic e-commerce operations database.
-- Chosen for realistic join depth (customers -> orders -> order_items -> products),
-- a self-join (employees.manager_id), and natural aggregation/filter/subquery targets.

CREATE TABLE customers (
    customer_id     INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    email           TEXT NOT NULL UNIQUE,
    country         TEXT NOT NULL,
    signup_date     TEXT NOT NULL,      -- ISO date
    tier            TEXT NOT NULL CHECK (tier IN ('free', 'silver', 'gold', 'platinum'))
);

CREATE TABLE employees (
    employee_id     INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    role            TEXT NOT NULL,
    hire_date       TEXT NOT NULL,
    manager_id      INTEGER REFERENCES employees(employee_id)
);

CREATE TABLE products (
    product_id      INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    category        TEXT NOT NULL,
    price           REAL NOT NULL,
    stock_quantity  INTEGER NOT NULL
);

CREATE TABLE orders (
    order_id        INTEGER PRIMARY KEY,
    customer_id     INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date      TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('pending', 'shipped', 'delivered', 'cancelled', 'returned'))
);

CREATE TABLE order_items (
    order_item_id   INTEGER PRIMARY KEY,
    order_id        INTEGER NOT NULL REFERENCES orders(order_id),
    product_id      INTEGER NOT NULL REFERENCES products(product_id),
    quantity        INTEGER NOT NULL,
    unit_price      REAL NOT NULL       -- price at time of purchase, may differ from products.price
);

CREATE TABLE reviews (
    review_id       INTEGER PRIMARY KEY,
    product_id      INTEGER NOT NULL REFERENCES products(product_id),
    customer_id     INTEGER NOT NULL REFERENCES customers(customer_id),
    rating          INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    review_date     TEXT NOT NULL,
    comment         TEXT
);

CREATE TABLE support_tickets (
    ticket_id           INTEGER PRIMARY KEY,
    customer_id         INTEGER NOT NULL REFERENCES customers(customer_id),
    order_id            INTEGER REFERENCES orders(order_id),
    assigned_employee_id INTEGER REFERENCES employees(employee_id),
    status              TEXT NOT NULL CHECK (status IN ('open', 'in_progress', 'resolved', 'closed')),
    priority            TEXT NOT NULL CHECK (priority IN ('low', 'medium', 'high', 'urgent')),
    created_date        TEXT NOT NULL
);

CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_order_items_order ON order_items(order_id);
CREATE INDEX idx_order_items_product ON order_items(product_id);
CREATE INDEX idx_reviews_product ON reviews(product_id);
CREATE INDEX idx_tickets_customer ON support_tickets(customer_id);
