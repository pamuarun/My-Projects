create database coal;
use coal;
select * from coal;
drop table coal;

#1st moment - Measure of Central Tendency

#Mean
SELECT AVG(RB_4800kcal_fob) AS mean_column
FROM coal;
SELECT AVG(RB_5500kcal_fob) AS mean_column1
FROM coal;
SELECT AVG(RB_5700kcal_fob) AS mean_column2
FROM coal;
SELECT AVG(RB_6000kcal_avg) AS mean_column3
FROM coal;
SELECT AVG(India_5500kcal_cfr) AS mean_column4
FROM coal;

-- Median for RB_4800kcal_fob
SELECT RB_4800kcal_fob AS median_RB_4800kcal_fob
FROM (
    SELECT RB_4800kcal_fob, 
           ROW_NUMBER() OVER (ORDER BY RB_4800kcal_fob) AS row_num,
           COUNT(*) OVER () AS total_count
    FROM coal
    WHERE RB_4800kcal_fob IS NOT NULL
) AS subquery
WHERE row_num = FLOOR((total_count + 1) / 2) 
   OR row_num = FLOOR((total_count + 2) / 2);

-- Median for RB_5500kcal_fob
SELECT RB_5500kcal_fob AS median_RB_5500kcal_fob
FROM (
    SELECT RB_5500kcal_fob, 
           ROW_NUMBER() OVER (ORDER BY RB_5500kcal_fob) AS row_num,
           COUNT(*) OVER () AS total_count
    FROM coal
    WHERE RB_5500kcal_fob IS NOT NULL
) AS subquery
WHERE row_num = FLOOR((total_count + 1) / 2) 
   OR row_num = FLOOR((total_count + 2) / 2);

-- Median for RB_5700kcal_fob
SELECT RB_5700kcal_fob AS median_RB_5700kcal_fob
FROM (
    SELECT RB_5700kcal_fob, 
           ROW_NUMBER() OVER (ORDER BY RB_5700kcal_fob) AS row_num,
           COUNT(*) OVER () AS total_count
    FROM coal
    WHERE RB_5700kcal_fob IS NOT NULL
) AS subquery
WHERE row_num = FLOOR((total_count + 1) / 2) 
   OR row_num = FLOOR((total_count + 2) / 2);

-- Median for RB_6000kcal_avg
SELECT RB_6000kcal_avg AS median_RB_6000kcal_avg
FROM (
    SELECT RB_6000kcal_avg, 
           ROW_NUMBER() OVER (ORDER BY RB_6000kcal_avg) AS row_num,
           COUNT(*) OVER () AS total_count
    FROM coal
    WHERE RB_6000kcal_avg IS NOT NULL
) AS subquery
WHERE row_num = FLOOR((total_count + 1) / 2) 
   OR row_num = FLOOR((total_count + 2) / 2);

-- Median for India_5500kcal_cfr
SELECT India_5500kcal_cfr AS median_India_5500kcal_cfr
FROM (
    SELECT India_5500kcal_cfr, 
           ROW_NUMBER() OVER (ORDER BY India_5500kcal_cfr) AS row_num,
           COUNT(*) OVER () AS total_count
    FROM coal
    WHERE India_5500kcal_cfr IS NOT NULL
) AS subquery
WHERE row_num = FLOOR((total_count + 1) / 2) 
   OR row_num = FLOOR((total_count + 2) / 2);


-- Mode for RB_4800kcal_fob
SELECT RB_4800kcal_fob AS mode_RB_4800kcal_fob
FROM (
    SELECT RB_4800kcal_fob, COUNT(*) AS frequency
    FROM coal
    WHERE RB_4800kcal_fob IS NOT NULL
    GROUP BY RB_4800kcal_fob
    ORDER BY frequency DESC
    LIMIT 1
) AS subquery;

-- Mode for RB_5500kcal_fob
SELECT RB_5500kcal_fob AS mode_RB_5500kcal_fob
FROM (
    SELECT RB_5500kcal_fob, COUNT(*) AS frequency
    FROM coal
    WHERE RB_5500kcal_fob IS NOT NULL
    GROUP BY RB_5500kcal_fob
    ORDER BY frequency DESC
    LIMIT 1
) AS subquery;

-- Mode for RB_5700kcal_fob
SELECT RB_5700kcal_fob AS mode_RB_5700kcal_fob
FROM (
    SELECT RB_5700kcal_fob, COUNT(*) AS frequency
    FROM coal
    WHERE RB_5700kcal_fob IS NOT NULL
    GROUP BY RB_5700kcal_fob
    ORDER BY frequency DESC
    LIMIT 1
) AS subquery;

-- Mode for RB_6000kcal_avg
SELECT RB_6000kcal_avg AS mode_RB_6000kcal_avg
FROM (
    SELECT RB_6000kcal_avg, COUNT(*) AS frequency
    FROM coal
    WHERE RB_6000kcal_avg IS NOT NULL
    GROUP BY RB_6000kcal_avg
    ORDER BY frequency DESC
    LIMIT 1
) AS subquery;

-- Mode for India_5500kcal_cfr
SELECT India_5500kcal_cfr AS mode_India_5500kcal_cfr
FROM (
    SELECT India_5500kcal_cfr, COUNT(*) AS frequency
    FROM coal
    WHERE India_5500kcal_cfr IS NOT NULL
    GROUP BY India_5500kcal_cfr
    ORDER BY frequency DESC
    LIMIT 1
) AS subquery;


#Measure of Dispersion - 2nd Moment
#Standard Deviation
SELECT 
    STDDEV(RB_4800kcal_fob) AS stddev_RB_4800kcal_fob,
    STDDEV(RB_5500kcal_fob) AS stddev_RB_5500kcal_fob,
    STDDEV(RB_5700kcal_fob) AS stddev_RB_5700kcal_fob,
    STDDEV(RB_6000kcal_avg) AS stddev_RB_6000kcal_avg,
    STDDEV(India_5500kcal_cfr) AS stddev_India_5500kcal_cfr
FROM coal;

#Range
SELECT 
    MAX(RB_4800kcal_fob) - MIN(RB_4800kcal_fob) AS range_RB_4800kcal_fob,
    MAX(RB_5500kcal_fob) - MIN(RB_5500kcal_fob) AS range_RB_5500kcal_fob,
    MAX(RB_5700kcal_fob) - MIN(RB_5700kcal_fob) AS range_RB_5700kcal_fob,
    MAX(RB_6000kcal_avg) - MIN(RB_6000kcal_avg) AS range_RB_6000kcal_avg,
    MAX(India_5500kcal_cfr) - MIN(India_5500kcal_cfr) AS range_India_5500kcal_cfr
FROM coal;

#Variance
SELECT 
    VARIANCE(RB_4800kcal_fob) AS variance_RB_4800kcal_fob,
    VARIANCE(RB_5500kcal_fob) AS variance_RB_5500kcal_fob,
    VARIANCE(RB_5700kcal_fob) AS variance_RB_5700kcal_fob,
    VARIANCE(RB_6000kcal_avg) AS variance_RB_6000kcal_avg,
    VARIANCE(India_5500kcal_cfr) AS variance_India_5500kcal_cfr
FROM coal;

#Skewness - 3rd Business Moment
-- Skewness for RB_4800kcal_fob
SELECT (
    SUM(POWER(RB_4800kcal_fob - (SELECT AVG(RB_4800kcal_fob) FROM coal), 3)) /
    (COUNT(*) * POWER((SELECT STDDEV(RB_4800kcal_fob) FROM coal), 3))
) AS skewness_RB_4800kcal_fob
FROM coal
WHERE RB_4800kcal_fob IS NOT NULL;

-- Skewness for RB_5500kcal_fob
SELECT (
    SUM(POWER(RB_5500kcal_fob - (SELECT AVG(RB_5500kcal_fob) FROM coal), 3)) /
    (COUNT(*) * POWER((SELECT STDDEV(RB_5500kcal_fob) FROM coal), 3))
) AS skewness_RB_5500kcal_fob
FROM coal
WHERE RB_5500kcal_fob IS NOT NULL;

-- Skewness for RB_5700kcal_fob
SELECT (
    SUM(POWER(RB_5700kcal_fob - (SELECT AVG(RB_5700kcal_fob) FROM coal), 3)) /
    (COUNT(*) * POWER((SELECT STDDEV(RB_5700kcal_fob) FROM coal), 3))
) AS skewness_RB_5700kcal_fob
FROM coal
WHERE RB_5700kcal_fob IS NOT NULL;

-- Skewness for RB_6000kcal_avg
SELECT (
    SUM(POWER(RB_6000kcal_avg - (SELECT AVG(RB_6000kcal_avg) FROM coal), 3)) /
    (COUNT(*) * POWER((SELECT STDDEV(RB_6000kcal_avg) FROM coal), 3))
) AS skewness_RB_6000kcal_avg
FROM coal
WHERE RB_6000kcal_avg IS NOT NULL;

-- Skewness for India_5500kcal_cfr
SELECT (
    SUM(POWER(India_5500kcal_cfr - (SELECT AVG(India_5500kcal_cfr) FROM coal), 3)) /
    (COUNT(*) * POWER((SELECT STDDEV(India_5500kcal_cfr) FROM coal), 3))
) AS skewness_India_5500kcal_cfr
FROM coal
WHERE India_5500kcal_cfr IS NOT NULL;

#Kurtosis - 4th Moment
-- Kurtosis for RB_4800kcal_fob
SELECT (
    (SUM(POWER(RB_4800kcal_fob - (SELECT AVG(RB_4800kcal_fob) FROM coal), 4)) /
    (COUNT(*) * POWER((SELECT STDDEV(RB_4800kcal_fob) FROM coal), 4))) - 3
) AS kurtosis_RB_4800kcal_fob
FROM coal
WHERE RB_4800kcal_fob IS NOT NULL;

-- Kurtosis for RB_5500kcal_fob
SELECT (
    (SUM(POWER(RB_5500kcal_fob - (SELECT AVG(RB_5500kcal_fob) FROM coal), 4)) /
    (COUNT(*) * POWER((SELECT STDDEV(RB_5500kcal_fob) FROM coal), 4))) - 3
) AS kurtosis_RB_5500kcal_fob
FROM coal
WHERE RB_5500kcal_fob IS NOT NULL;

-- Kurtosis for RB_5700kcal_fob
SELECT (
    (SUM(POWER(RB_5700kcal_fob - (SELECT AVG(RB_5700kcal_fob) FROM coal), 4)) /
    (COUNT(*) * POWER((SELECT STDDEV(RB_5700kcal_fob) FROM coal), 4))) - 3
) AS kurtosis_RB_5700kcal_fob
FROM coal
WHERE RB_5700kcal_fob IS NOT NULL;

-- Kurtosis for RB_6000kcal_avg
SELECT (
    (SUM(POWER(RB_6000kcal_avg - (SELECT AVG(RB_6000kcal_avg) FROM coal), 4)) /
    (COUNT(*) * POWER((SELECT STDDEV(RB_6000kcal_avg) FROM coal), 4))) - 3
) AS kurtosis_RB_6000kcal_avg
FROM coal
WHERE RB_6000kcal_avg IS NOT NULL;

-- Kurtosis for India_5500kcal_cfr
SELECT (
    (SUM(POWER(India_5500kcal_cfr - (SELECT AVG(India_5500kcal_cfr) FROM coal), 4)) /
    (COUNT(*) * POWER((SELECT STDDEV(India_5500kcal_cfr) FROM coal), 4))) - 3
) AS kurtosis_India_5500kcal_cfr
FROM coal
WHERE India_5500kcal_cfr IS NOT NULL;

drop table forecast_results;
select * from forecast_results;