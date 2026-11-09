CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT,
    signup_date TEXT
);

CREATE TABLE tasks (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    done INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    amount REAL NOT NULL,
    created_at TEXT NOT NULL
);

INSERT INTO users (id, name, country, signup_date) VALUES
    (1, 'Alice', 'CN', '2026-01-01'),
    (2, 'Bob', 'US', '2026-01-02'),
    (3, 'Carol', 'JP', '2026-02-01'),
    (4, 'Dave', 'CN', '2026-03-01');

INSERT INTO tasks (id, user_id, title, done) VALUES
    (1, 1, 'write report', 1),
    (2, 1, 'review pr', 0),
    (3, 1, 'plan sprint', 1),
    (4, 2, 'deploy service', 1),
    (5, 3, 'design ui', 0),
    (6, 3, 'write docs', 1);

INSERT INTO orders (id, user_id, amount, created_at) VALUES
    (1, 1, 120.50, '2026-01-10'),
    (2, 1, 80.00, '2026-02-11'),
    (3, 2, 300.00, '2026-01-05'),
    (4, 3, 45.25, '2025-12-31');
