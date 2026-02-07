-- Create a separate database for our project data
-- (Airflow uses the 'airflow' database, we use 'market_analytics')
-- Tables are created by SQLAlchemy (see src/database/models.py)

CREATE DATABASE market_analytics;
