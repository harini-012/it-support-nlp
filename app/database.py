import os

import psycopg2
from dotenv import load_dotenv


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured.")

    return psycopg2.connect(DATABASE_URL)


def create_tables():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id SERIAL PRIMARY KEY,

            title TEXT NOT NULL,

            description TEXT,

            predicted_category VARCHAR(100),

            predicted_priority VARCHAR(50),

            assigned_team VARCHAR(100),

            category_confidence FLOAT,

            priority_confidence FLOAT,

            suggested_solution TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()

    cursor.close()
    connection.close()


def save_ticket(
    title,
    description,
    prediction
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO tickets (
            title,
            description,
            predicted_category,
            predicted_priority,
            assigned_team,
            category_confidence,
            priority_confidence,
            suggested_solution
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            title,
            description,
            prediction["category"],
            prediction["priority"],
            prediction["assigned_team"],
            prediction["category_confidence"],
            prediction["priority_confidence"],
            prediction["suggested_solution"]
        )
    )

    ticket_id = cursor.fetchone()[0]

    connection.commit()

    cursor.close()
    connection.close()

    return ticket_id


def get_tickets():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            title,
            description,
            predicted_category,
            predicted_priority,
            assigned_team,
            category_confidence,
            priority_confidence,
            suggested_solution,
            created_at
        FROM tickets
        ORDER BY created_at DESC
        """
    )

    rows = cursor.fetchall()

    cursor.close()
    connection.close()

    tickets = []

    for row in rows:

        tickets.append({
            "id": row[0],
            "title": row[1],
            "description": row[2],
            "category": row[3],
            "priority": row[4],
            "assigned_team": row[5],
            "category_confidence": row[6],
            "priority_confidence": row[7],
            "suggested_solution": row[8],
            "created_at": row[9].isoformat()
        })

    return tickets