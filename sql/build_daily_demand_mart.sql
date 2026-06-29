DROP TABLE IF EXISTS daily_demand_mart;
CREATE TABLE daily_demand_mart AS
SELECT
    series_id,
    date_idx,
    category,
    store,
    sales,
    demand,
    stockout,
    promo,
    price,
    recovered_demand_mean,
    drat_state,
    drat_entropy,
    LAG(sales, 1) OVER (PARTITION BY series_id ORDER BY date_idx) AS sql_lag_1,
    AVG(sales) OVER (
        PARTITION BY series_id
        ORDER BY date_idx
        ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
    ) AS sql_roll_mean_7
FROM training_panel;
