# Energy Price Forecasting & Monte Carlo Simulation

## Advanced Machine Learning, Energy Market Dynamics and Probabilistic Price Forecasting Through 10,000 Monte Carlo Scenarios

An integrated Machine Learning and stochastic simulation project combining electricity price forecasting, statistical feature optimization and Monte Carlo analysis to investigate the relationships between energy prices, production, consumption, environmental conditions and market uncertainty.

## Project Overview

### Objective: From Electricity Price Prediction to Probabilistic Decision Support

The project develops a quantitative framework for forecasting electricity prices based on the interaction of energy production, system consumption, weather conditions and temporal factors.

Electricity prices are influenced by multiple interconnected variables, including national electricity demand, wind generation, temperature, wind speed, CO₂ intensity and variations in consumption patterns throughout the day and across different periods.

The objective is not limited to predicting an individual price value.

The project combines **predictive Machine Learning with stochastic scenario simulation** to investigate the variability of electricity prices under different combinations of market and environmental conditions.

The framework addresses four analytical objectives:

- **Price formation analysis:** identifying the relationships between electricity prices, energy demand, production and environmental factors.
- **Predictive optimization:** improving forecasting performance through statistical feature selection, model comparison and hyperparameter optimization.
- **Uncertainty quantification:** estimating the distribution of possible electricity prices across 10,000 simulated scenarios.
- **Decision support:** providing quantitative evidence relevant to energy procurement, financial planning, price risk assessment and market strategy.

The distinctive contribution is the integration of a trained predictive model with Monte Carlo simulations, transforming a point-estimation approach into a probabilistic analysis of energy price variability.

## Analytical Framework

The methodology integrates three complementary components.

**Statistical Analysis and Feature Engineering**

A multivariate analysis investigates correlations and multicollinearity to identify the relationships among energy market variables and optimize the predictive feature set.

**Machine Learning and Model Optimization**

Several regression algorithms are evaluated to identify the model most capable of capturing the relationships between energy market variables and actual electricity prices.

**Monte Carlo Simulation**

The optimized predictive model is applied to 10,000 simulated combinations of explanatory variables, generating a distribution of possible electricity prices and associated percentile ranges.

## Methodology

### 1. Energy Market Data and Price Formation Variables

The project uses an electricity market dataset containing historical observations of electricity prices, production, system demand, weather conditions and temporal characteristics.

The target variable is:

**SMPEP2 — Actual Electricity Price**

The explanatory variables initially include several categories.

**Energy Production and Demand**
- ForecastWindProduction — Forecast wind energy production.
- ActualWindProduction — Actual wind energy production.
- SystemLoadEA — Forecast national electricity system load.
- SystemLoadEP2 — Actual national electricity system load.

**Energy Prices**
- SMPEA — Forecast electricity price.
- SMPEP2 — Actual electricity price, used as the prediction target.

**Environmental Conditions**
- ORKTemperature — Measured temperature.
- ORKWindspeed — Measured wind speed.
- CO2Intensity — Actual CO₂ intensity associated with electricity generation.

**Temporal Variables**
- DayOfWeek.
- WeekOfYear.
- Day.
- Month.
- Year.
- PeriodOfDay.
- HolidayFlag.

The dataset allows the analysis of electricity price formation as the combined result of supply, demand, environmental and temporal conditions.

### 2. Advanced Exploratory Data Analysis

The project investigates the statistical characteristics of electricity market variables, focusing on relationships relevant to price formation and predictive modeling.

The analysis includes:

- Statistical distribution analysis.
- Kernel Density Estimation (KDE) to investigate distribution shapes.
- Skewness and kurtosis analysis.
- Comparison of observed and forecast energy variables.
- Examination of relationships between electricity prices and explanatory factors.

Particular attention is given to asymmetric distributions and extreme observations in energy market variables.

These characteristics are relevant because electricity prices and production conditions can exhibit irregular behavior that complicates predictive modeling.

### 3. Correlation Analysis and Multicollinearity Optimization

A distinctive methodological component is the combined use of **Pearson correlation analysis and the Variance Inflation Factor (VIF)**.

Pearson correlation is used to investigate pairwise relationships among explanatory variables and between those variables and electricity prices.

The initial analysis identifies several strongly correlated pairs:

| Variable Pair | Pearson Correlation |
|---|---:|
| WeekOfYear / Month | 0.97 |
| SystemLoadEA / SystemLoadEP2 | 0.97 |
| ForecastWindProduction / ActualWindProduction | 0.95 |
| ForecastWindProduction / ORKWindspeed | 0.71 |

The project extends this investigation through VIF analysis, which measures the degree of multicollinearity among explanatory variables.

**Selected initial VIF results:**

| Variable | VIF |
|---|---:|
| SystemLoadEA | 34.29 |
| SystemLoadEP2 | 31.68 |
| ActualWindProduction | 12.93 |
| ForecastWindProduction | 11.80 |
| WeekOfYear | 18.76 |
| Month | 18.79 |

These results reveal substantial redundancy among variables describing related energy production, consumption and temporal phenomena.

The project applies feature selection to reduce multicollinearity while retaining variables with relevant predictive information.

The final feature set prioritizes forecast electricity prices, actual system load, environmental conditions and temporal characteristics.

Following feature optimization, the notebook reports improved correlation with the prediction target for two important variables:

| Variable | Before | After |
|---|---:|---:|
| Forecast Electricity Price (SMPEA) | 0.62 | 0.68 |
| Actual System Load (SystemLoadEP2) | 0.52 | 0.62 |

The resulting dataset provides a more focused analytical foundation for model training.

### 4. Statistical Feature Transformation

The project applies differentiated transformations according to the statistical characteristics of the variables.

The methodology includes:

- Logarithmic transformation of forecast electricity prices.
- Yeo-Johnson transformations for temperature and wind speed.
- Standardization of environmental, load and temporal variables.

These transformations address differences in variable scales and distribution asymmetry.

The approach combines statistical diagnostics with targeted feature engineering rather than applying uniform transformations indiscriminately.

### 5. Comparative Machine Learning Modeling

The project evaluates multiple regression algorithms to identify the most effective approach for electricity price forecasting.

The models investigated include:

- Random Forest Regressor.
- Gradient Boosting Regressor.
- Ridge Regression.
- Support Vector Regression with an RBF kernel.

The comparative analysis examines predictive performance through Mean Squared Error (MSE) and the coefficient of determination (R²).

The initial Random Forest model achieves:

| Metric | Result |
|---|---:|
| R² | 0.7782 |
| MSE | 0.03699 |

Alternative regression approaches are evaluated to investigate whether different model structures can improve predictive performance.

The notebook reports that Gradient Boosting, Ridge Regression and Support Vector Regression do not outperform the Random Forest baseline.

The analysis therefore retains Random Forest as the preferred modeling approach.

### 6. Random Forest Hyperparameter Optimization

The Random Forest model is subsequently optimized using **RandomizedSearchCV with five-fold cross-validation**.

The hyperparameter search evaluates alternative configurations involving:

- Number of decision trees.
- Maximum tree depth.
- Minimum samples required for splitting.
- Minimum samples per leaf.
- Number of features considered at each split.
- Bootstrap sampling.

The search evaluates 20 sampled hyperparameter combinations, using R² as the optimization metric.

**Selected optimized configuration:**

| Hyperparameter | Value |
|---|---:|
| Number of estimators | 300 |
| Maximum depth | 20 |
| Minimum samples split | 2 |
| Minimum samples leaf | 1 |
| Maximum features | log2 |
| Bootstrap | False |

**Optimized model performance:**

| Metric | Initial Random Forest | Optimized Random Forest |
|---|---:|---:|
| R² | 0.7782 | **0.8037** |
| MSE | 0.03699 | **0.03274** |

The optimized model explains approximately **80.37% of the observed target variability** in the test dataset.

This represents an improvement over the initial Random Forest model and provides the predictive foundation for the subsequent stochastic simulation.

The reported results refer to the transformed target scale used in the notebook.

### 7. Monte Carlo Simulation: 10,000 Electricity Price Scenarios

The central component of the project is the integration of the optimized Random Forest model with **Monte Carlo simulation**.

Rather than relying exclusively on a single predicted electricity price, the framework generates 10,000 different combinations of explanatory variables.

Each scenario represents an alternative configuration of the factors influencing electricity price formation.

The simulation includes variations in:

- Forecast electricity prices.
- Temperature.
- Wind speed.
- CO₂ intensity.
- Actual electricity system load.
- Day of the week.
- Day of the month.
- Month.
- Time of day.

Continuous variables are simulated using normal distributions parameterized by their training-data means and standard deviations.

Temporal variables are sampled from their defined ranges, while the simulation year is fixed at 2025.

The trained Random Forest model then predicts the electricity price associated with each simulated scenario.

This produces **10,000 model-generated price estimates**, forming the basis of the probabilistic analysis.

The approach provides a way to investigate the sensitivity of predicted prices to alternative combinations of energy demand, weather conditions and temporal characteristics.

### 8. Probabilistic Price Distribution and Quantile Analysis

The Monte Carlo results are analyzed through a histogram of simulated electricity prices and the calculation of selected distribution quantiles.

The notebook reports the following simulation results:

| Quantile | Predicted Price |
|---|---:|
| 5th percentile | 3.7899 |
| 25th percentile | 3.9859 |
| 50th percentile — Median | **4.0974** |
| 75th percentile | 4.2794 |
| 95th percentile | 4.4601 |

The quantiles provide a structured view of the distribution of predicted electricity prices under the simulated conditions.

For example:

- The median represents the central value of the simulated price distribution.
- The 25th and 75th percentiles describe the central 50% of simulated outcomes.
- The 5th and 95th percentiles provide a wider range encompassing approximately 90% of the simulated outcomes.

These are **scenario-distribution quantiles**, not formally calibrated prediction intervals for future realized market prices.

The values are expressed on the target scale used by the trained model.

This probabilistic representation offers a more informative perspective than a single point estimate, allowing price variability to be examined across different combinations of market and environmental factors.

## Financial and Strategic Decision-Making Applications

### Energy Procurement and Cost Planning

The framework provides an analytical foundation for investigating possible electricity price outcomes under alternative demand and environmental conditions.

Probabilistic estimates can support the assessment of procurement costs and potential deviations from expected expenditure.

### Financial Budgeting and Scenario Analysis

The Monte Carlo distribution can inform scenario-based financial planning by providing a range of model-generated electricity price outcomes.

This is particularly relevant when energy expenditure represents a significant component of operating costs.

### Price Risk Management

Quantile analysis helps characterize the variability of predicted prices across simulated scenarios.

The framework can support preliminary assessments of exposure to adverse price conditions and motivate more detailed stress testing or hedging analysis.

### Energy Market Strategy

The combination of feature analysis, predictive modeling and simulation helps investigate how energy prices respond to changes in production, consumption and environmental conditions.

These insights may inform market monitoring, strategic planning and energy-related financial decisions.

### Uncertainty-Aware Decision Support

The principal advantage of the framework is its ability to extend predictive analysis beyond a single expected value.

By generating multiple scenarios, the methodology offers a structured way to explore uncertainty and the range of outcomes associated with different input conditions.

## Key Findings

**1. Energy Prices Reflect Interconnected Market Drivers**

The analysis identifies meaningful relationships between electricity prices, forecast prices, national system load, environmental conditions and temporal factors.

**2. Multicollinearity Analysis Improves Feature Selection**

Combining Pearson correlation and VIF helps identify redundant explanatory variables and supports the construction of a more focused predictive dataset.

**3. Random Forest Provides the Strongest Evaluated Predictive Performance**

Among the regression models tested, Random Forest delivers the best reported performance.

**4. Hyperparameter Optimization Improves Model Fit**

The optimized Random Forest achieves an R² of 0.8037 and an MSE of 0.03274.

**5. Monte Carlo Simulation Extends the Model into Probabilistic Analysis**

The simulation generates 10,000 alternative input scenarios and produces a distribution of predicted electricity prices.

**6. Quantile Analysis Provides a Structured View of Price Variability**

The simulated distribution provides central and wider percentile ranges that can support uncertainty-aware analysis.

## Technologies and Methodologies

**Programming and Data Processing**

Python · Pandas · NumPy

**Statistical Analysis**

SciPy · Statsmodels · Pearson Correlation · Variance Inflation Factor · Skewness and Kurtosis Analysis

**Feature Engineering**

Logarithmic Transformation · Yeo-Johnson Transformation · Standardization · Multicollinearity Reduction · Feature Selection

**Machine Learning**

Scikit-learn · Random Forest Regression · Gradient Boosting · Ridge Regression · Support Vector Regression

**Model Optimization**

RandomizedSearchCV · Five-Fold Cross-Validation · Hyperparameter Optimization

**Stochastic Modeling**

Monte Carlo Simulation · Scenario Generation · Probability Distribution Analysis · Quantile Estimation

**Visualization**

Matplotlib · Seaborn · KDE Plots · Correlation Heatmaps · Simulation Histograms

**Development Environment**

Google Colab · Jupyter Notebook

## Project Files

- [Electricity_Price_Forecasting_Model.ipynb](Electricity_Price_Forecasting_Model.ipynb) — Complete analytical notebook, including statistical analysis, feature selection, comparative modeling, Random Forest optimization and Monte Carlo simulation.

### Dataset

The project uses the electricity market dataset stored in this repository:

[datasets/electricity.csv](datasets/electricity.csv)

The notebook retrieves the dataset directly from GitHub:

https://raw.githubusercontent.com/Valeria-Raele/energy-price-forecasting-monte-carlo/refs/heads/main/datasets/electricity.csv

Original dataset source:

https://github.com/amankharwal/Website-data/blob/master/electricity.csv

The dataset is loaded programmatically, without requiring manual file uploads during notebook execution.

## Analytical Workflow

**Energy Market Data → Statistical Analysis → Correlation and VIF Assessment → Feature Engineering → Comparative Machine Learning → Random Forest Optimization → 10,000 Monte Carlo Scenarios → Probabilistic Price Distribution → Financial and Strategic Decision Support.**

The distinctive contribution of the project is the integration of statistical feature optimization, predictive Machine Learning and stochastic simulation within a single analytical framework.

The methodology connects the analysis of energy market fundamentals with probabilistic price estimation, extending the practical relevance of forecasting to financial planning and uncertainty assessment.

## Project Scope and Limitations

Independent quantitative analysis project focused on electricity price forecasting and variability estimation through Machine Learning and Monte Carlo simulation.

The project demonstrates how statistical analysis, predictive optimization and stochastic modeling can be combined to investigate electricity price formation and support data-driven decisions.

The analysis is experimental and based on historical observations. The simulation uses independently sampled input variables and therefore does not fully preserve the historical dependencies among energy market drivers. It should be interpreted as exploratory scenario analysis rather than a validated probabilistic forecasting system.

The notebook uses a random train-test split rather than chronological out-of-sample validation. Further validation would be required before operational deployment or use in financial risk management.
