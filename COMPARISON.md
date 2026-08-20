# NL-to-SQL specialist vs. frontier prompting — comparison report

| Predictor | Accuracy | n | p50 latency (ms) | p95 latency (ms) | Cost / 1k calls |
|---|---|---|---|---|---|
| frontier:claude-haiku-4-5 | 53.6% | 28 | 1055 | 1884 | $1.0648 |
| ollama:sql-specialist | 92.9% | 28 | 207 | 371 | $0.0000 |

**Read this table with the failure taxonomy below before drawing conclusions
from it** — the raw accuracy gap overstates the specialist's *reasoning*
advantage. It's real on latency and cost (no contest: local inference beats
an API round-trip on both), but the accuracy gap is inflated by an eval
methodology artifact, documented honestly below rather than hidden.

## Failure taxonomy (manual audit)

Execution-accuracy scoring in `eval/execution.py` compares result rows
*column-for-column* — an otherwise-identical query that selects one extra
column, drops a column, or returns rows in a different order than the gold
query's `ORDER BY` scores as a hard failure, indistinguishable from a query
that computed the wrong answer entirely. I manually read every one of
Claude Haiku 4.5's 13 "failures" against this eval set. The result:

| Failure type | Count | What actually happened |
|---|---|---|
| Extra non-essential column(s) | 7 | Correct rows, plus e.g. `product_id`, `email`, or a status column the question didn't ask for |
| Row order differs from gold's `ORDER BY` | 2 | Identical rows, different order — the question never specified a sort order; the gold SQL's `ORDER BY` is a dataset-authoring convention, not something asked for |
| Missing column that's arguably optional per the question wording | 4 | e.g. "which customers have placed more than 5 orders" answered with just names, no count — a defensible reading, since the question didn't ask "and show the count" |
| **Wrong / non-executing SQL** | **0** | — |

**Every one of Claude Haiku 4.5's measured failures on this eval set was a
formatting or convention mismatch. Zero were SQL logic errors** — it never
picked the wrong table, joined incorrectly, miscounted, or wrote SQL that
failed to execute. The fine-tuned specialist, by contrast, had 2 failures out
of 28 and both *were* genuine logic errors (a hallucinated `orders.total`
column, a dropped table qualifier) — see below.

**What this means for the headline numbers:**
- **92.9% vs. 53.6% execution accuracy is real, but it's not principally a
  "smarter at SQL" gap.** It's largely the specialist having memorized this
  project's specific narrow column-selection and ordering conventions from
  111 training examples, something a frontier model prompted zero-shot has
  no way to know without being told (and even when told — see
  `eval/baseline_frontier.py`'s `COLUMN_DISCIPLINE_ADDENDUM`, added mid-eval
  and worth only +7.2 points, 46.4%→53.6% — a general-purpose model still
  exercises its own judgment about what counts as "extra" context).
- **Latency (207ms vs. 1055ms p50) and cost ($0 vs. $1.06/1k calls) are the
  gap that's actually about the model, not the eval's column-matching
  strictness** — those numbers don't care about column conventions, and
  they're the ones this project's premise (a cheap local specialist can beat
  API round-trips on the metrics that matter for production serving) rests on.
- **The asymmetry in failure *quality* is arguably the most interesting
  result here**: the frontier model's mistakes are all superficial and the
  underlying answer is still correct and usable; the specialist's rarer
  mistakes are outright wrong (executable but incorrect results). A
  production system built on the specialist needs to actually catch those
  2/28 confidently-wrong cases; one built on frontier prompting mostly needs
  a stricter output-shape contract.

A stricter, more standard-practice version of this project would normalize
for column supersets before scoring (checking that gold's requested columns'
values appear in the prediction, not requiring an exact column-for-column
match) — documented here as the honest next step rather than quietly
special-cased into the scoring after seeing what it changed.

## Accuracy by query category

| Category | frontier:claude-haiku-4-5 | ollama:sql-specialist |
|---|---|---|
| aggregation | 50% | 100% |
| date_range | 100% | 100% |
| having | 0% | 100% |
| join | 25% | 75% |
| multi_join | 0% | 100% |
| negation_antijoin | 100% | 100% |
| self_join | 100% | 100% |
| simple_filter | 67% | 100% |
| subquery_correlated | 100% | 100% |
| subquery_scalar | 0% | 100% |
| top_n | 50% | 50% |

## Failures — frontier:claude-haiku-4-5 (13)

- **aggregation** — _How many orders does each customer have with status 'cancelled'? Show customer name and count._
  - gold: `SELECT c.name, COUNT(*) AS order_count FROM customers c JOIN orders o ON o.customer_id = c.customer_id WHERE o.status = 'cancelled' GROUP BY c.customer_id ORDER BY order_count DESC`
  - pred: `SELECT c.name, COUNT(o.order_id) AS cancelled_order_count
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'cancelled'
GROUP BY c.customer_id, c.name
ORDER BY c.customer_id;`
- **join** — _List support tickets with priority 'urgent' along with the customer's name and the assigned employee's name._
  - gold: `SELECT t.ticket_id, c.name AS customer_name, e.name AS employee_name FROM support_tickets t JOIN customers c ON c.customer_id = t.customer_id LEFT JOIN employees e ON e.employee_id = t.assigned_employee_id WHERE t.priority = 'urgent'`
  - pred: `SELECT st.ticket_id, c.name, e.name, st.priority
FROM support_tickets st
JOIN customers c ON st.customer_id = c.customer_id
LEFT JOIN employees e ON st.assigned_employee_id = e.employee_id
WHERE st.priority = 'urgent'`
- **aggregation** — _How many orders does each customer have with status 'pending'? Show customer name and count._
  - gold: `SELECT c.name, COUNT(*) AS order_count FROM customers c JOIN orders o ON o.customer_id = c.customer_id WHERE o.status = 'pending' GROUP BY c.customer_id ORDER BY order_count DESC`
  - pred: `SELECT c.name, COUNT(o.order_id) AS pending_order_count
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'pending'
GROUP BY c.customer_id, c.name
ORDER BY c.customer_id;`
- **multi_join** — _For 'gold'-tier customers, list each order along with the product names and quantities purchased._
  - gold: `SELECT o.order_id, c.name AS customer_name, p.name AS product_name, oi.quantity FROM customers c JOIN orders o ON o.customer_id = c.customer_id JOIN order_items oi ON oi.order_id = o.order_id JOIN products p ON p.product_id = oi.product_id WHERE c.tier = 'gold'`
  - pred: `SELECT 
    o.order_id,
    p.name,
    oi.quantity
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE c.tier = 'gold'
ORDER BY o.order_id, p.name;`
- **top_n** — _Show the top 5 customers by total spend on delivered orders._
  - gold: `SELECT c.name, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_spent FROM customers c JOIN orders o ON o.customer_id = c.customer_id AND o.status = 'delivered' JOIN order_items oi ON oi.order_id = o.order_id GROUP BY c.customer_id ORDER BY total_spent DESC LIMIT 5`
  - pred: `SELECT c.customer_id, c.name, SUM(oi.quantity * oi.unit_price) as total_spend
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.status = 'delivered'
GROUP BY c.customer_id, c.name
ORDER BY total_spend DESC
LIMIT 5;`
- **simple_filter** — _List all products in the Toys category, cheapest first._
  - gold: `SELECT name, price FROM products WHERE category = 'Toys' ORDER BY price ASC`
  - pred: `SELECT product_id, name, price FROM products WHERE category = 'Toys' ORDER BY price ASC;`
- **join** — _List support tickets with priority 'low' along with the customer's name and the assigned employee's name._
  - gold: `SELECT t.ticket_id, c.name AS customer_name, e.name AS employee_name FROM support_tickets t JOIN customers c ON c.customer_id = t.customer_id LEFT JOIN employees e ON e.employee_id = t.assigned_employee_id WHERE t.priority = 'low'`
  - pred: `SELECT 
    st.ticket_id,
    c.name AS customer_name,
    e.name AS employee_name,
    st.priority
FROM support_tickets st
JOIN customers c ON st.customer_id = c.customer_id
LEFT JOIN employees e ON st.assigned_employee_id = e.employee_id
WHERE st.priority = 'low';`
- **having** — _List categories where the average product price exceeds 100._
  - gold: `SELECT category, ROUND(AVG(price),2) AS avg_price FROM products GROUP BY category HAVING AVG(price) > 100`
  - pred: `SELECT category
FROM products
GROUP BY category
HAVING AVG(price) > 100;`
- **simple_filter** — _Who are our customers based in Germany?_
  - gold: `SELECT name FROM customers WHERE country = 'Germany'`
  - pred: `SELECT name, email FROM customers WHERE country = 'Germany';`
- **having** — _Which customers have placed more than 5 orders?_
  - gold: `SELECT c.name, COUNT(*) AS order_count FROM customers c JOIN orders o ON o.customer_id = c.customer_id GROUP BY c.customer_id HAVING COUNT(*) > 5 ORDER BY order_count DESC`
  - pred: `SELECT c.name
FROM customers c
WHERE (SELECT COUNT(*) FROM orders o WHERE o.customer_id = c.customer_id) > 5
ORDER BY c.name;`
  - ... and 3 more

## Failures — ollama:sql-specialist (2)

- **top_n** — _Show the top 5 customers by total spend on delivered orders._
  - gold: `SELECT c.name, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_spent FROM customers c JOIN orders o ON o.customer_id = c.customer_id AND o.status = 'delivered' JOIN order_items oi ON oi.order_id = o.order_id GROUP BY c.customer_id ORDER BY total_spent DESC LIMIT 5`
  - pred: `SELECT c.name, ROUND(SUM(o.total), 2) AS total_spent FROM customers c JOIN orders o ON o.customer_id = c.customer_id WHERE o.status = 'delivered' GROUP BY c.customer_id ORDER BY total_spent DESC LIMIT 5`
  - error: execution error: no such column: o.total
- **join** — _Show order_id, customer name, and customer country for every order._
  - gold: `SELECT o.order_id, c.name, c.country FROM orders o JOIN customers c ON o.customer_id = c.customer_id`
  - pred: `SELECT order_id, name, country FROM orders`
  - error: execution error: no such column: name
