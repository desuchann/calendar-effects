# Calendar Effects in Stock Markets

An empirical investigation into if well-known calendar anomalies produce statistically significant patterns in equity market returns.

## Overview

This project examines four calendar effects using daily returns from the **FTSE 100** and **S&P 500**:

- 🎄 January Effect
- 🎃 Halloween Effect ("Sell in May and Go Away")
- 🔁 Turn-of-the-Month Effect
- 📆 Day-of-the-Week Effect

The study evaluates whether these anomalies remain detectable in modern financial markets and whether their strength changes across different sample periods.

## Methodology

- Downloaded historical market data from **Yahoo Finance**
- Calculated daily logarithmic returns
- Constructed calendar dummy variables
- Estimated Ordinary Least Squares (OLS) regression models
- Applied **Newey-West (HAC)** standard errors to account for heteroskedasticity and autocorrelation
- Compared results across:
  - FTSE 100 vs S&P 500
  - Pre-2008 vs Post-2008 samples
  - Baseline and alternative monthly model specifications

## Key Findings

- The **Turn-of-the-Month effect** showed the strongest evidence, particularly before 2008.
- Little statistical support was found for the **January**, **Halloween**, or **Day-of-the-Week** effects.
- Most detected anomalies weakened considerably in more recent data, supporting the view that market efficiency evolves over time.

## Technologies

- Python
- pandas
- NumPy
- statsmodels
- matplotlib
- Jupyter Notebook

## Skills Demonstrated

- Financial data analysis
- Econometric modelling
- Time series analysis
- Statistical hypothesis testing
- Regression analysis
- Data visualisation
