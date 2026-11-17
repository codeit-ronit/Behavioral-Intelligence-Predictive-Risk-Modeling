SELECT t.Account AS account, t.date, COUNT(*) AS trade_count,
          SUM(t."Size USD") AS volume, AVG(t."Size USD") AS avg_size,
          SUM(t."Closed PnL") AS daily_pnl,
          SUM(CASE WHEN t."Closed PnL" > 0 THEN 1 ELSE 0 END) AS wins,
          SUM(CASE WHEN t."Closed PnL" != 0 THEN 1 ELSE 0 END) AS realized_trades,
          SUM(t.is_long)*1.0/(SUM(t.is_short)+1) AS long_short_ratio,
          AVG(t."Closed PnL"*t."Closed PnL") - AVG(t."Closed PnL")*AVG(t."Closed PnL") AS pnl_variance,
          s.value AS sentiment_value, COALESCE(s.classification, 'Unknown') AS sentiment_category,
          s.date AS sentiment_date
          FROM trades t LEFT JOIN sentiment s ON s.date = date(t.date, '-1 day')
          GROUP BY t.Account, t.date, s.value, s.classification, s.date
          ORDER BY t.Account, t.date;
