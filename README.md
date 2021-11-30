# Natural Language Processing for Trending Stocks and Cryptos
## Monash University FinTech Bootcamp Project 2
### Using sentiment score on crypto asset to predict direction of price change

We want to google trends api to search for trending stocks and crypto 
We want to analyse trending stocks and cryptos in various forums such as news api, reddit and twitter
We also want to analyse previous large price movements and do sentiment analysis on the previous week of data


Methodology
- Find dates in trading days where there was a significant price movement on a crypto asset, i.e 10% gain or loss
- Put in a dataframe.
- Add class to dataframe such that: => 10% price movement = 1, < 10% but greater > -10% = 0,  
- Collect prior 3 -10 days social media commentary, articles, news and online contents on the crypto assest and put in a dataframe.
- Use TF-IDF to analyse text in dataframe to check for unique words and their relevancy. Words with high TF-IDF score > 0.04 will be stored as an associated keyword for the crypto asset. 
- Perform sentiment analysis for cryto asset on the online content before the 10% price change
- Train ML model on the sentiments scores across the social media channels using class as our target variable.
- Predict target variable on test data
- Test ML model on other crypto assets
- 
- Repeat above steps on other crypto assets 
- Find buzzword associated with each asset and affects price movement.
- Use ML model to predict when a 10% price movement is likely to occure on a given the current sentiment score on the crypto asset.
