# EnergyForecasting
The Goal:
To predict house portfolio consumption

The Project Scope contains:
1) Basic EDA for the daily and hourly consumption per sub sample
2) House consumption correlation and clustering
3) Basic Feature Engineering
4) Modelling (Martingale process, Linear Regression and LightGBM)

The best Model given small sample size is:
1) Linear Regression on Daily Series + Decomposition on hourly series based on Avg ratio
2) LightGBM on daily does outperform Linear model for some little nmae score difference, but this small difference is not justifiable to choose complicated model 

What needs to be improved:
1) Increase the number of samples (houses)
2) Increase the sample size of time series
3) Experiment with feature engineering particularly with cloudness, humidity and temperature  (for the first two no transformations were applied)
4) Collect more data about the location of each house and type (clustering shows that there exists different consumption pattern)
5) Create better feature selection process (currently is missing)
6) Create more appropriate buckets for each category (for instance using Anova tests)
7) Source more data such as solar panel, number of residents etc..
8) Build N models for each cluster  (preferably on daily data since it is less noisy)
9) Build more advanced decomposition to hourly data, rather than using simple avg ratio.
As example, one can remove yearly (summer, fall, winter, spring) seasonality and build time series model ARIMA and then
aggregate it with model which predicts daily consumption

