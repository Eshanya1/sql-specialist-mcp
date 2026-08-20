# NL-to-SQL specialist vs. frontier prompting — comparison report

| Predictor | Accuracy | n | p50 latency (ms) | p95 latency (ms) | Cost / 1k calls |
|---|---|---|---|---|---|
| ollama:sql-specialist | 92.9% | 28 | 207 | 371 | $0.0000 |

## Accuracy by query category

| Category | ollama:sql-specialist |
|---|---|
| aggregation | 100% |
| date_range | 100% |
| having | 100% |
| join | 75% |
| multi_join | 100% |
| negation_antijoin | 100% |
| self_join | 100% |
| simple_filter | 100% |
| subquery_correlated | 100% |
| subquery_scalar | 100% |
| top_n | 50% |

## Failures — ollama:sql-specialist (2)

- **top_n** — _Show the top 5 customers by total spend on delivered orders._
  - gold: `SELECT c.name, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_spent FROM customers c JOIN orders o ON o.customer_id = c.customer_id AND o.status = 'delivered' JOIN order_items oi ON oi.order_id = o.order_id GROUP BY c.customer_id ORDER BY total_spent DESC LIMIT 5`
  - pred: `SELECT c.name, ROUND(SUM(o.total), 2) AS total_spent FROM customers c JOIN orders o ON o.customer_id = c.customer_id WHERE o.status = 'delivered' GROUP BY c.customer_id ORDER BY total_spent DESC LIMIT 5`
  - error: execution error: no such column: o.total
- **join** — _Show order_id, customer name, and customer country for every order._
  - gold: `SELECT o.order_id, c.name, c.country FROM orders o JOIN customers c ON o.customer_id = c.customer_id`
  - pred: `SELECT order_id, name, country FROM orders`
  - error: execution error: no such column: name
