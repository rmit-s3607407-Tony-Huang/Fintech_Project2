# Natural Language Processing for Trending Stocks and Cryptos
## Monash University FinTech Bootcamp Project 2


We want to google trends api to search for trending stocks and crypto 
We want to analyse trending stocks and cryptos in various forums such as news api, reddit and twitter
We also want to analyse previous large price movements and do sentiment analysis on the previous week of data


Methodology
- Find dates where there was a significant price movement on MANA, i.e 10% gain or loss
- Use NLP to analyse previous 30 days keyword trend for the stock or crypto asset or keywords associated with the asset i.e NFT, cross chain, metaverse etc to measure the strength of sentiments on asset/associated keywords before the price movement
- Repeat analysis DOT, BTC, ETH & LINK with significant price movements.
- Find buzzword associated with each asset and affects price movements
- Testing sentiment periods of 1D, 2D & 5D (apply weighting based on results)
- Find a threshold sentiment score that translates into a significant movement in asset price. Use score as benchmark 
- Select new crypto for future predictions and analyse their current trend.
- Analyse sentiment score for crypto asset and compare with the model benchmark
- Make predictions on the probable future direction. 
