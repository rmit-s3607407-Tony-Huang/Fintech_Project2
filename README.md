# Natural Language Processing for Trending Stocks and Cryptos
## Monash University FinTech Bootcamp Project 2
### Using sentiment score on crypto asset to predict direction of price change

## Hypothesis
Public sentiments on crypto assets can predict future price movement.


## Strategy
We will use NLP to mine text and measure sentiments on popular social media forums - reddit and search engines- google on days before a signifcant price movement. Our analysis will be looking for a correlation between sentiment score and crypto asset price movement.



## Methodology
- Using trading api, find dates in trading days where there was a significant price movement on a crypto asset, i.e 10% gain or loss
- Put in a dataframe.
- Add class to dataframe such that: => 10% price movement = 1, < 10% but greater > -10% = 0,  <= -10% = -1
- Collect prior 3 -10 days social media commentary, articles, news and online contents on the crypto assest and put in a dataframe.
- Use TF-IDF to analyse text in dataframe to check for unique words and their relevancy. Words with high TF-IDF score > 0.04 will be stored as an associated keyword for the crypto asset. 
- Perform sentiment analysis for cryto asset on the online content before the 10% price change
- Train ML model on the sentiments scores across the social media channels using class as our target variable.
- Predict target variable on test data
- Repeat above steps on other crypto assets 
- Use ML model to predict when a 10% price movement is likely to occur given the current sentiment score on the crypto asset.
