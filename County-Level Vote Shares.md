---
title: "County-Level Vote Shares"
date: "May 10, 2025"
author: "Isaac Stiepleman"
output: pdf_document
fontsize: 11pt
geometry: margin = 1in
---
# Introduction

Problem Statement: Accurately predicting Democratic vote share across Washington counties is critical for effective allocation of campaign resources and targeted voter outreach strategies.

Audience: This analysis specifically benefits the Washington State Democratic Party. By identifying key demographic predictors and clustering counties based on demographic profiles, campaign strategists can efficiently allocate resources to regions with the highest potential impact.



# Data Sources


``` r
#Libraries
suppressPackageStartupMessages({
  library(tidyverse)
  library(readr)
  library(readxl)
  library(dplyr)
  library(ggplot2)
  library(caret)})

#Datasets
election_24 <- read_csv("allcounties.csv")
demographic <- read_excel("county_demos.xlsx", sheet = "Total")
```

Election Results: I used official Washington State election data from 2024, containing county-level total votes and Democratic vote shares of every contest on the ballot (I will limit my analysis to just the Presidential race). 

Demographic Data: The Washington Office of Financial Management (OFM) provided detailed county-level population breakdowns across multiple age brackets, gender, and racial categories from 2020-2024 (I will only use 2024).

# Data Building


``` r
#Vote data
election_pres <- election_24 |>
  filter(Race == "United States President/Vice President") #just want president

demvote_24 <- election_pres |>
  filter(grepl("Democrat", Party, ignore.case = TRUE)) |>
  group_by(County) |>
  summarise(dem_votes = sum(Votes))

vote_24 <- election_pres |>
  filter(!is.na(Party)) |>
  group_by(County) |>
  summarise(total_votes = sum(Votes))

votedata_24 <- left_join(demvote_24, vote_24, by = "County") |>
  mutate(dem_share = dem_votes / total_votes)

#Demographic Data
population_totals <- demographic |> 
  rename(County = `Area Name`) |>
  filter(County != "Washington", Year == 2024, `Age Group` == "Total") |> 
  select(County, Total)

biodata_24 <- demographic |>
  rename(County = `Area Name`) |>
  filter(County != "Washington" & Year == 2024) |>
  filter(`Age Group` != "Total") |>
  pivot_longer(cols = -c(County, `Area ID`, Year, `Age Group`),
               names_to = "Demo",
               values_to = "Count") |>
  unite("ColName", `Age Group`, Demo, sep = "_") |>
  pivot_wider(names_from = ColName,
              values_from = Count) |>
  select(-contains("Total"), -`Area ID`, -Year) |>
  left_join(population_totals, by = "County") |>
  mutate(across(-County, as.numeric)) |>
  mutate(across(-c(County, Total), ~ .x / Total))

#Merge
modeldata_24 <- left_join(votedata_24, biodata_24, by = "County")
head(modeldata_24[1:6])
```

```
## # A tibble: 6 x 6
##   County  dem_votes total_votes dem_share `0-4_Male` `0-4_Female`
##   <chr>       <dbl>       <dbl>     <dbl>      <dbl>        <dbl>
## 1 Adams        1455        5345     0.272     0.0389       0.0396
## 2 Asotin       4082       11396     0.358     0.0247       0.0226
## 3 Benton      37662       99947     0.377     0.0303       0.0291
## 4 Chelan      18397       41893     0.439     0.0254       0.0233
## 5 Clallam     25440       48390     0.526     0.0187       0.0176
## 6 Clark      143206      274795     0.521     0.0271       0.0259
```

``` r
#To make callbacks easier
demo_cols <- grep("^[0-9]", names(modeldata_24), value = TRUE)

#Missingness Summary
missing_summary <- modeldata_24 %>%
  summarise(across(everything(), ~ sum(is.na(.)) / n(), .names = "{col}")) %>%
  pivot_longer(everything(), names_to = "feature", values_to = "pct_missing")

sum(missing_summary$pct_missing)
```

```
## [1] 0
```

## Feature Engineering
Age Proportions: I engineered demographic features by calculating the proportion of each age group relative to county total populations.

- Rationale: Age and racial demographics are strong indicators of political participation and partisan preferences, crucial for predicting vote share and preferences at the county level.

Data Cleaning: There are no missing data points across the datasets. All 39 counties provided complete demographic and electoral information.

# Data Visualization


``` r
#Histogram of top 8 features by demographic variance across counties
top_features <- modeldata_24 %>%
  select(all_of(demo_cols)) |>
  summarise(across(everything(), var)) %>%
  pivot_longer(everything(), names_to = "feature", values_to = "variance") %>%
  arrange(desc(variance)) %>%
  slice_head(n = 8) %>%
  pull(feature)

modeldata_24 %>%
  select(all_of(top_features)) %>%
  pivot_longer(everything(), names_to = "feature", values_to = "value") %>%
  ggplot(aes(x = value)) + 
    geom_histogram(bins = 30) +
    facet_wrap(~ feature, scales = "free", ncol = 2) +
    labs(title = "Distribution of Highest-Variance Demographic Features",
         x = "Proportion of County Population",
         y = "Count of Counties") +
    theme_minimal()
```



\begin{center}\includegraphics{County-Level-Vote-Shares_files/figure-latex/unnamed-chunk-3-1} \end{center}

Age Distributions: Histograms of the top 8 age/gender/race proportions demonstrated notable variance across counties. Serves as an example of the features that are most likely to drive differences in Democratic vote share.

# Baseline Model


``` r
racial_groups <- c("White", "Black", "AIAN", "Asian", "NHPI", "Two or More Races")

corrs <- sapply(racial_groups,
                function(grp) {
                  columns <- grep(grp, names(modeldata_24), value = TRUE)
                  pct <- rowSums(modeldata_24[, columns])
                  cor(pct, modeldata_24$dem_share)
                }); corrs
```

```
##             White             Black              AIAN             Asian 
##        -0.4381819         0.5137249        -0.2352918         0.5947475 
##              NHPI Two or More Races 
##         0.4525263         0.1342507
```

``` r
best <- names(which.max(abs(corrs))) #group w/ highest correlation to dem_share
best_cols <- grep(best, names(modeldata_24), value = TRUE)
modeldata_24$best_pct <- rowSums(modeldata_24[, best_cols])

baseline <- lm(dem_share ~ best_pct, data = modeldata_24)
summary(baseline)
```

```
## 
## Call:
## lm(formula = dem_share ~ best_pct, data = modeldata_24)
## 
## Residuals:
##      Min       1Q   Median       3Q      Max 
## -0.16313 -0.07314  0.00174  0.04897  0.32427 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)   0.3838     0.0221   17.37  < 2e-16 ***
## best_pct      1.9379     0.4306    4.50 6.53e-05 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.1081 on 37 degrees of freedom
## Multiple R-squared:  0.3537,	Adjusted R-squared:  0.3363 
## F-statistic: 20.25 on 1 and 37 DF,  p-value: 6.529e-05
```

``` r
sqrt(mean((modeldata_24$dem_share - predict(baseline))^2)) #rmse
```

```
## [1] 0.1052982
```

Baseline Model: A single-predictor OLS model (using the racial group most correlated with `dem_share`) yielded an $R^2$ of $0.35$ and an RMSE of approximately $0.11$, indicating sluggish predictive power.

# K-Means Clusters


``` r
set.seed(2)

#Test k from 1 to 20
scaled_24 <- scale(modeldata_24[, demo_cols])
ratio <- numeric(20)

for (k in 1:20) {
  km <- kmeans(scaled_24, centers = k)
  ratio[k] <- km$betweenss / km$tot.withinss
}

results <- data.frame(k = 1:20, Ratio = ratio)

ggplot(results, aes(x = k, y = Ratio)) +
  geom_line() +
  labs(title = "Ratio (BetweenSS / Total WithinSS) by Clusters") +
  scale_x_continuous(breaks = 1:20) +
  theme_minimal()
```



\begin{center}\includegraphics{County-Level-Vote-Shares_files/figure-latex/unnamed-chunk-5-1} \end{center}

``` r
km <- kmeans(scaled_24, centers = 7)
modeldata_24$cluster <- km$cluster

ggplot(modeldata_24, aes(x = factor(cluster), y = dem_share)) +
  geom_boxplot(fill = "steelblue") +
  labs(title = "Democratic Vote Share by Cluster",
       x = "Cluster", y = "Democratic Vote Share") +
  theme_minimal()
```



\begin{center}\includegraphics{County-Level-Vote-Shares_files/figure-latex/unnamed-chunk-5-2} \end{center}

``` r
#Cluster Summary
cluster_summary <- modeldata_24 %>%
  group_by(cluster) %>%
  summarise(across(matches("^[0-9]"), ~ mean(.)),
            mean_dem_share = mean(dem_share))

head(cluster_summary)
```

```
## # A tibble: 6 x 254
##   cluster `0-4_Male` `0-4_Female` `0-4_White Male` `0-4_White Female`
##     <int>      <dbl>        <dbl>            <dbl>              <dbl>
## 1       1     0.0240       0.0230           0.0153             0.0148
## 2       2     0.0162       0.0143           0.0116             0.0107
## 3       3     0.0193       0.0194           0.0145             0.0143
## 4       4     0.0333       0.0327           0.0235             0.0236
## 5       5     0.0245       0.0236           0.0182             0.0176
## 6       6     0.0242       0.0207           0.0190             0.0166
## # i 249 more variables: `0-4_Black Male` <dbl>, `0-4_Black Female` <dbl>,
## #   `0-4_AIAN Male` <dbl>, `0-4_AIAN Female` <dbl>, `0-4_Asian Male` <dbl>,
## #   `0-4_Asian Female` <dbl>, `0-4_NHPI Male` <dbl>, `0-4_NHPI Female` <dbl>,
## #   `0-4_Two or More Races Male` <dbl>, `0-4_Two or More Races Female` <dbl>,
## #   `5-9_Male` <dbl>, `5-9_Female` <dbl>, `5-9_White Male` <dbl>,
## #   `5-9_White Female` <dbl>, `5-9_Black Male` <dbl>, `5-9_Black Female` <dbl>,
## #   `5-9_AIAN Male` <dbl>, `5-9_AIAN Female` <dbl>, `5-9_Asian Male` <dbl>, ...
```

K-Means Clustering: Since the ratio rose steadily with no clear elbow, I selected 7 clusters as a midpoint balance (since the curve spikes at 14). Clustering revealed distinct demographic profiles across clusters, with notable differences in Democratic vote share (boxplots showed clear cluster differentiation).

# Evaluation (PCA + OLS)


``` r
#PCA
pca <- prcomp(scaled_24)

modeldata_24$pca1 <- pca$x[,1]
modeldata_24$pca2 <- pca$x[,2]

ggplot(modeldata_24, aes(x = pca1, y = pca2, color = factor(cluster))) +
  geom_point(size = 3) +
    labs(title = "PCA Scores Plot",
       x = "PC1", y = "PC2", color = "Cluster") +
  theme_minimal()
```



\begin{center}\includegraphics{County-Level-Vote-Shares_files/figure-latex/unnamed-chunk-6-1} \end{center}

``` r
#Inspect Loadings
pca_loadings <- pca$rotation[, 1:2]
head(pca_loadings)
```

```
##                           PC1         PC2
## 0-4_Male         -0.045403031 -0.10613541
## 0-4_Female       -0.051630782 -0.10644409
## 0-4_White Male   -0.008289269 -0.13780115
## 0-4_White Female -0.015866197 -0.14068557
## 0-4_Black Male   -0.091810630  0.05889510
## 0-4_Black Female -0.090337368  0.05825289
```

``` r
#OLS
cumul_var <- cumsum((pca$sdev^2) / sum(pca$sdev^2))
k <- which(cumul_var > 0.9)[1]
pca_data <- data.frame(pca$x[, 1:k])

ols <- lm(modeldata_24$dem_share ~ ., data = pca_data)
summary(ols)
```

```
## 
## Call:
## lm(formula = modeldata_24$dem_share ~ ., data = pca_data)
## 
## Residuals:
##       Min        1Q    Median        3Q       Max 
## -0.110458 -0.022514 -0.001813  0.023432  0.170608 
## 
## Coefficients:
##               Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  0.4456156  0.0094390  47.210  < 2e-16 ***
## PC1         -0.0062545  0.0009947  -6.288 1.18e-06 ***
## PC2          0.0113465  0.0016100   7.048 1.75e-07 ***
## PC3         -0.0036345  0.0016652  -2.183  0.03829 *  
## PC4         -0.0031112  0.0020922  -1.487  0.14902    
## PC5         -0.0010399  0.0028523  -0.365  0.71838    
## PC6         -0.0167286  0.0033326  -5.020 3.19e-05 ***
## PC7         -0.0010593  0.0037424  -0.283  0.77937    
## PC8          0.0120695  0.0039258   3.074  0.00491 ** 
## PC9         -0.0232766  0.0044652  -5.213 1.92e-05 ***
## PC10         0.0026952  0.0048973   0.550  0.58678    
## PC11        -0.0052587  0.0050554  -1.040  0.30781    
## PC12        -0.0146786  0.0055634  -2.638  0.01388 *  
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.05895 on 26 degrees of freedom
## Multiple R-squared:  0.865,	Adjusted R-squared:  0.8027 
## F-statistic: 13.88 on 12 and 26 DF,  p-value: 2.176e-08
```

Principal Component Analysis (PCA): PCA substantially reduced feature complexity.

PCA-based Regression: I retained the first 12 PCs (enough to explain $90\%$ of variance) and regressed Democratic vote share on them. My model achieved an $R^2$ of around $0.87$, significantly reducing prediction error compared to the baseline.

# Cross-Validation


``` r
set.seed(123)  
cv_ctrl <- trainControl(method = "cv", number = 5)

lm_cv <- train(dem_share ~ .,
               data = modeldata_24 |> 
                 select(dem_share, all_of(demo_cols)),
               method = "lm",
               preProcess = c("center", "scale", "pca"),
               trControl = cv_ctrl,
               metric = "RMSE")

print(lm_cv)
```

```
## Linear Regression 
## 
##  39 samples
## 252 predictors
## 
## Pre-processing: centered (252), scaled (252), principal component
##  signal extraction (252) 
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 31, 31, 32, 31, 31 
## Resampling results:
## 
##   RMSE        Rsquared   MAE       
##   0.07555665  0.7496601  0.05885723
## 
## Tuning parameter 'intercept' was held constant at a value of TRUE
```

Cross-Validation: Five-fold cross-validation provided a robust assessment of generalization, achieving a mean cross-validated $R^2$ of approximately $0.75$ and an RMSE of about $0.08$, affirming good model performance beyond training data.

Comparative Analysis: The PCA-based regression model outperformed the trivial baseline predictor, validating the effectiveness and added predictive value of demographic-derived features.

# Conclusions

## Strategic insights:

(Clusters ranked by Democratic vote share)

- Cluster 7 captures dense, highly diverse urban centers with the strongest Democratic support ($59.2\%$). These are the Puget Sound metro counties with the largest and most diverse populations. This cluster typifies progressive strongholds—think major cities like Seattle where multiracial communities thrive, and liberal voting patterns dominate.
- Cluster 2 counties also show a very high average level of Democratic support ($57.0\%$). Demographically, these counties have fewer very young children (ages 0-4) and slightly less racial diversity. Campaign resources should be invested here to maximize Democratic vote share through targeted outreach, emphasizing policies appealing to older and established voter bases.
- Cluster 3 is another majority Democratic area ($52.8\%$) which is moderately diverse, indicating potential battleground status. Campaign strategy should aim to consolidate and potentially expand Democratic support through inclusive and moderate policy messaging.
- Cluster 5 counties exhibit a moderate Democratic share (approximately $46.3\%$), but also demonstrate strong recruiting potential. Given their demographic similarity to Cluster 3, resources should similarly target inclusive, broad-based messaging and turnout initiatives.
- Clusters 1 and 4 show substantially lower Democratic performance (approximately $36.6\%$ and $34.8\%$ respectively). These counties tend to have higher proportions of very young children. Campaign investment in these counties might require longer-term community engagement strategies rather than short-term voter-turnout initiatives.
- Cluster 6 represents low-population areas that are older and super-majority White. Unsurprisingly they deliver a dismal Democratic vote share ($31.6\%$). It's a poor fit for short-term spending aimed at improving Democratic turnout.

## PCA Insights:

- PC1 (37% of variance) contrasts younger, more diverse counties with older, whiter ones. Its loadings are strongest on residents under about 55 and on Asian, Black, and NHPI residents, and have the opposite sign for residents aged 65 or older and for White and American Indian and Alaska Native residents. Its correlation with Democratic vote share is -0.45, so lower PC1 scores (younger, more diverse counties) go with higher Democratic share.
- PC2 (14% of variance) is a mixed age-and-race axis, with older residents (55+) and Asian, Black, and NHPI residents on one side and White and multiracial residents on the other. Its correlation with Democratic vote share is 0.51. I’d avoid giving it a single demographic label.

## Limitations

- Geographic Granularity: County-level data masks within-county variation. Try to leverage precinct-level data in the future.
- Additional Data Needs: Using field outreach metrics would further enhance predictive accuracy and actionability.
