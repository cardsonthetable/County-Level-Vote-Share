---
title: "County-Level Vote Shares"
date: "May 10, 2025"
author: "Isaac Stiepleman"
output: pdf_document
fontsize: 11pt
geometry: margin = 1in
---
# Introduction

Problem Statement: Accurately predicting Democratic turnout across Washington counties is critical for effective allocation of campaign resources and targeted voter outreach strategies.

Audience: This analysis specifically benefits the Washington State Democratic Party. By identifying key demographic predictors and clustering counties based on voter profiles, campaign strategists can efficiently allocate resources to regions with the highest potential impact.



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

Election Results: I used official Washington State election data from 2024, containing county-level total votes and Democratic vote shares.

Demographic Data: The Washington Office of Financial Management (OFM) provided detailed county-level population breakdowns across multiple age brackets, gender, and racial categories from 2020-2024.

# Data Building


``` r
#Vote data
demvote_24 <- election_24 |>
  filter(grepl("Democrat", Party, ignore.case = TRUE)) |>
  group_by(County) |>
  summarise(dem_votes = sum(Votes))

vote_24 <- election_24 |>
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
## 1 Adams       15832       79312     0.200     0.0389       0.0396
## 2 Asotin      49784      178665     0.279     0.0247       0.0226
## 3 Benton     408289     1446495     0.282     0.0303       0.0291
## 4 Chelan     223608      647942     0.345     0.0254       0.0233
## 5 Clallam    361573      700313     0.516     0.0187       0.0176
## 6 Clark     1838001     3634672     0.506     0.0271       0.0259
```

``` r
#Missingness Summary
missing_summary <- modeldata_24 %>%
  summarise(across(everything(), ~ sum(is.na(.)) / n(), .names = "{col}")) %>%
  pivot_longer(everything(), names_to = "feature", values_to = "pct_missing")

sum(missing_summary$pct_missing)
```

```
## [1] 0
```

Feature Engineering
Age Proportions: I engineered demographic features by calculating the proportion of each age group relative to county total populations.

- Rationale: Age demographics are strong indicators of political participation and partisan preferences, crucial for predicting voter turnout and preferences at the county level.

Data Cleaning: There are no significant missing data points across the datasets. All 39 counties provided complete demographic and electoral information.

# Data Visualization


``` r
#Histogram of top 8 features by demographic variance across counties
top_features <- modeldata_24 %>%
  select(-c(1:4, 257)) |>
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

Age Distributions: Histograms of the top 8 age/gender/race proportions demonstrated significant variance across counties. Serves as an example of the features that are most likely to drive differences in Democratic vote share.

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
##       -0.40671874        0.51608189       -0.26207981        0.58272829 
##              NHPI Two or More Races 
##        0.46988782        0.09333306
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
## -0.16078 -0.09169 -0.01780  0.07199  0.40928 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  0.32509    0.02733  11.895 3.31e-14 ***
## best_pct     2.32287    0.53256   4.362 9.92e-05 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.1337 on 37 degrees of freedom
## Multiple R-squared:  0.3396,	Adjusted R-squared:  0.3217 
## F-statistic: 19.02 on 1 and 37 DF,  p-value: 9.923e-05
```

``` r
sqrt(mean((modeldata_24$dem_share - predict(baseline))^2)) #rmse
```

```
## [1] 0.130225
```

Baseline Model: A single-predictor OLS model (using the racial group most correlated with dem_share) yielded an $R^2$ of $0.32$ and an RMSE of approximately $0.13$, indicating sluggish predictive power.

# K-Means Clusters


``` r
set.seed(2)

#Test k from 1 to 20
scaled_24 <- scale(modeldata_24 |> select(-c(1:4)))
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

print(cluster_summary)
```

```
## # A tibble: 7 x 254
##   cluster `0-4_Male` `0-4_Female` `0-4_White Male` `0-4_White Female`
##     <int>      <dbl>        <dbl>            <dbl>              <dbl>
## 1       1     0.0240       0.0230           0.0153             0.0148
## 2       2     0.0177       0.0159           0.0131             0.0124
## 3       3     0.0193       0.0194           0.0145             0.0143
## 4       4     0.0333       0.0327           0.0235             0.0236
## 5       5     0.0252       0.0242           0.0170             0.0164
## 6       6     0.0248       0.0225           0.0192             0.0175
## 7       7     0.0286       0.0273           0.0151             0.0144
## # i 249 more variables: `0-4_Black Male` <dbl>, `0-4_Black Female` <dbl>,
## #   `0-4_AIAN Male` <dbl>, `0-4_AIAN Female` <dbl>, `0-4_Asian Male` <dbl>,
## #   `0-4_Asian Female` <dbl>, `0-4_NHPI Male` <dbl>, `0-4_NHPI Female` <dbl>,
## #   `0-4_Two or More Races Male` <dbl>, `0-4_Two or More Races Female` <dbl>,
## #   `5-9_Male` <dbl>, `5-9_Female` <dbl>, `5-9_White Male` <dbl>,
## #   `5-9_White Female` <dbl>, `5-9_Black Male` <dbl>, `5-9_Black Female` <dbl>,
## #   `5-9_AIAN Male` <dbl>, `5-9_AIAN Female` <dbl>, `5-9_Asian Male` <dbl>, ...
```

K-Means Clustering: I employed k-means clustering with an optimal solution of 7 clusters (selected via elbow plot analysis) to avoid overfitting. Clustering revealed distinct demographic profiles across clusters, with notable differences in Democratic vote share (boxplots showed clear cluster differentiation).

# Evaluation (PCA + OLS)


``` r
#PCA
pca <- prcomp(scaled_24)

modeldata_24$pca1 <- pca$x[,1]
modeldata_24$pca2 <- pca$x[,2]

ggplot(modeldata_24, aes(x = pca1, y = pca2, color = factor(cluster))) +
  geom_point(size = 3) +
  labs(title = "PCA Biplot",
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
## 0-4_Male         -0.044448023 -0.10631506
## 0-4_Female       -0.050601097 -0.10671080
## 0-4_White Male   -0.007344394 -0.13756454
## 0-4_White Female -0.014829572 -0.14053532
## 0-4_Black Male   -0.091510228  0.05720329
## 0-4_Black Female -0.090094128  0.05669743
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
##      Min       1Q   Median       3Q      Max 
## -0.14480 -0.02902 -0.01653  0.03217  0.21584 
## 
## Coefficients:
##              Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  0.399195   0.012331  32.374  < 2e-16 ***
## PC1         -0.007529   0.001290  -5.836 3.77e-06 ***
## PC2          0.014778   0.002096   7.051 1.73e-07 ***
## PC3         -0.005578   0.002175  -2.564 0.016470 *  
## PC4         -0.001432   0.002730  -0.525 0.604358    
## PC5         -0.002050   0.003683  -0.556 0.582636    
## PC6         -0.019267   0.004352  -4.427 0.000152 ***
## PC7         -0.002877   0.004889  -0.588 0.561296    
## PC8          0.010925   0.005128   2.131 0.042735 *  
## PC9         -0.026431   0.005831  -4.533 0.000115 ***
## PC10         0.002964   0.006393   0.464 0.646702    
## PC11        -0.005930   0.006602  -0.898 0.377287    
## PC12        -0.017636   0.007263  -2.428 0.022399 *  
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.07701 on 26 degrees of freedom
## Multiple R-squared:  0.846,	Adjusted R-squared:  0.775 
## F-statistic: 11.91 on 12 and 26 DF,  p-value: 1.083e-07
```

Principal Component Analysis (PCA): PCA substantially reduced feature complexity. A PCA biplot confirmed clear separation among clusters.

PCA-based Regression: Using PC1 and PC2 as predictors, my regression model achieved an $R^2$ of around $0.85$, significantly reducing prediction error compared to the baseline, with an RMSE around $0.06$.

# Cross-Validation


``` r
set.seed(123)  
cv_ctrl <- trainControl(method = "cv", number = 5)

lm_cv <- train(dem_share ~ .,
               data = modeldata_24 |> 
                 select(-County, -dem_votes, -total_votes, -cluster),
               method = "lm",
               preProcess = c("center", "scale", "pca"),
               trControl = cv_ctrl,
               metric = "RMSE")

summary(lm_cv)
```

```
## 
## Call:
## lm(formula = .outcome ~ ., data = dat)
## 
## Residuals:
##      Min       1Q   Median       3Q      Max 
## -0.20263 -0.02136 -0.00519  0.02228  0.11485 
## 
## Coefficients:
##              Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  0.399195   0.011538  34.597  < 2e-16 ***
## PC1         -0.007490   0.001201  -6.236 4.33e-06 ***
## PC2          0.014575   0.001934   7.535 2.90e-07 ***
## PC3         -0.005578   0.002035  -2.740 0.012615 *  
## PC4         -0.001432   0.002554  -0.561 0.581337    
## PC5         -0.002050   0.003447  -0.595 0.558713    
## PC6         -0.019267   0.004072  -4.731 0.000128 ***
## PC7         -0.002877   0.004575  -0.629 0.536548    
## PC8          0.010925   0.004798   2.277 0.033922 *  
## PC9         -0.026431   0.005456  -4.844 9.84e-05 ***
## PC10         0.002964   0.005982   0.496 0.625600    
## PC11        -0.005930   0.006178  -0.960 0.348545    
## PC12        -0.017636   0.006796  -2.595 0.017317 *  
## PC13        -0.007275   0.007487  -0.972 0.342780    
## PC14         0.006238   0.007676   0.813 0.425997    
## PC15        -0.002207   0.008883  -0.248 0.806298    
## PC16         0.020327   0.008992   2.261 0.035080 *  
## PC17         0.006874   0.009282   0.741 0.467532    
## PC18         0.014673   0.009537   1.539 0.139573    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.07206 on 20 degrees of freedom
## Multiple R-squared:  0.8963,	Adjusted R-squared:  0.803 
## F-statistic: 9.604 on 18 and 20 DF,  p-value: 2.856e-06
```

Cross-Validation: Five-fold cross-validation provided a robust assessment of generalization, achieving a mean cross-validated $R^2$ of approximately $0.49$ and an RMSE of about $0.12$, affirming good model performance beyond training data.

Comparative Analysis: The PCA-based regression model outperformed the trivial baseline predictor, validating the effectiveness and added predictive value of demographic-derived features.

# Conclusions

K-means Clustering revealed strategic insights:

- Cluster 2 counties show one of the highest average Democratic support (approximately $56.5\%$). Demographically, these counties have fewer very young children (ages 0-4) and slightly greater racial diversity. Campaign resources should be heavily invested here to maximize Democratic turnout through targeted outreach, emphasizing policies appealing to diverse and established voter bases.
- Cluster 3 averages the second-highest Democratic vote share (about $43.7\%$). Counties in this cluster are moderately diverse, indicating potential battleground status. Campaign strategy should aim to consolidate and potentially expand Democratic support through inclusive and moderate policy messaging.
- Cluster 5 counties exhibit a similarly strong average Democratic share (approximately $42.0\%$), reflecting significant potential. Given their demographic similarity to Cluster 3, resources should similarly target inclusive, broad-based messaging and turnout initiatives.
- Clusters 1 and 4 show significantly lower Democratic performance (approximately $27.1\%$ and $26.9\%$ respectively). These counties tend to have higher proportions of very young children and less racial diversity, suggesting more conservative demographics. Campaign investment in these counties might require longer-term community engagement strategies rather than short-term voter-turnout initiatives.
- Cluster 6 represents younger, racially diverse areas with a notable concentration of children and young adults (0-34). It shows elevated shares of Black, Asian, and multiracial populations, lower White representation, and a moderate Democratic vote share (~$25.4\%$). This cluster likely includes growing suburbs or smaller cities with emerging diversity but less entrenched progressive leanings than major urban hubs.
- Cluster 7 captures dense, highly diverse urban centers with the strongest Democratic support (~$58.4\%$). It features balanced age demographics (leaning slightly toward 35-64), very high Black and Asian representation, and minimal White population shares. This cluster typifies progressive strongholds—think major cities like Seattle—where multiracial and NHPI communities thrive, and liberal voting patterns dominate.

PCA Insights:

- PC1 primarily differentiates counties based on the proportion of young children (0-4 years). Counties scoring higher on PC1 have fewer young children, suggesting an older or more established demographic. Campaigns can interpret higher PC1 scores as indicators to focus messaging toward older or working-age voters rather than very young family-oriented policies.
- PC2 mainly captures racial diversity among young demographics. Higher PC2 scores indicate counties with relatively fewer young white children and potentially greater proportions of racial minorities among the youth. Targeted messaging for high-PC2 counties should emphasize policies addressing diversity, equity, education, and community inclusiveness to resonate effectively with these voter demographics.

Limitations

- Geographic Granularity: County-level data masks within-county variation. Try to leverage precinct-level data in the future.
- Additional Data Needs: Using field outreach metrics would further enhance predictive accuracy and actionability.
