# Natural Language Processing for Trending Stocks and Cryptos
## Monash University FinTech Bootcamp Project 2
### Using sentiment score on crypto asset to predict direction of price change

## Hypothesis
Public sentiments on crypto assets can predict future price movement.


## Strategy
We will use NLP to mine text and measure sentiments on popular social media forums (reddit) and search engine (google) on days before a signifcant price movement. Our analysis will be looking for a correlation between sentiment score and crypto asset price movement.



## Methodology
- Using trading api, find dates in trading days where there was a significant price movement on a crypto asset, i.e 10% gain or loss
- Put in a dataframe.
- Add class to dataframe such that: => 10% price movement = 1, < 10% but greater > -10% = 0,  <= -10% = -1
- Collect prior 3 -10 days social media commentary, articles, news and online contents on the crypto assest and put in a dataframe.
- Perform sentiment analysis for crypto asset on the online content before the 10% price change and store in the dataframe
- Use TF-IDF to analyse texts in the dataframe to check for unique words and their relevancy. Store Words with TF-IDF score > 0.04 as a feature in the crypto dataframe
- Train ML model on the sentiments and TF -IDF score across the social media channels using class as our target variable.
- Predict target variable on test data
- Repeat above steps on other crypto assets 
- Use ML model to predict when a 10% price movement is likely to occur given the current sentiment score on the crypto asset.

## Algorithms used
- KNN
- Random Forest
- XGBoost

## Findings
Google trend, google buzzword score and volume ranked higher than other features on both random forest and xgboost alogrithms

![image](https://user-images.githubusercontent.com/34574729/144804236-2a115c8c-af7a-48b7-8f2a-cb4880ffa51f.png)


## Conclusion
Based on the performance metrics observed in Random Forest and XGBoost, our analysis reveal that public sentiments does have effect on crypto price. However, our model is limited to data collected from Reddit and Google. 

## To Explore
-  Effect of price data on model performance
-  Data from other social media platforms and news articles i.e twitter, facebook, news api, etc
-  Volatility
