DROP TABLE IF EXISTS inventory_kpi_mart;
CREATE TABLE inventory_kpi_mart AS
SELECT
    model,
    series_id,
    COUNT(*) AS evaluated_days,
    SUM(demand) AS total_demand,
    SUM(sales) AS total_sales,
    SUM(lost_sales) AS total_lost_sales,
    SUM(cost) AS total_cost,
    AVG(ending_inventory) AS average_inventory,
    AVG(stockout) AS stockout_rate,
    1.0 - SUM(lost_sales) / NULLIF(SUM(demand), 0.0) AS fill_rate
FROM inventory_simulation
GROUP BY model, series_id;
