from asyncio import threads
import newspaper
from newspaper import Config, Article, Source, news_pool
import pandas as pd

config = Config()

config.fetch_images = False
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36"

config.browser_user_agent = USER_AGENT
config.request_timeout = 10
config.memoize_articles = False
config.language = "en"

news_df = pd.DataFrame()


cnn_paper = newspaper.build('http://cnn.com', config=config) # left

fox = newspaper.build('https://www.foxnews.com/', config=config) # right

bloomberg = newspaper.build('https://www.bloomberg.com/', config=config) # left-center

breitbart = newspaper.build('https://www.breitbart.com/', config=config) # right

cnbc = newspaper.build('https://www.cnbc.com/', config=config) # center

forbes = newspaper.build('https://www.forbes.com/', config=config) # center

huffpo = newspaper.build('https://www.huffpost.com/', config=config) # left

nypost = newspaper.build('https://nypost.com/', config=config) # right-center


news_pool_list = [cnn_paper, fox, bloomberg, breitbart, cnbc, forbes, huffpo, nypost]
'''
news_pool.set(news_pool_list, threads_per_source=2)
news_pool.join()
'''
for paper in news_pool_list:   
    print(paper.size())



def recover_articles(source, lean):

    print("Recover Article Started...")    
    article_title = []
    article_text = []
    article_date = []
    article_url = []
    article_alignment = []
    article_brand = []
    
    brand = source.brand

    for article in source.articles:
        try:
            article.download()
            article.parse()
        except:
            continue
        
        print(article)
        
        article_title.append(article.title)
        article_text.append(article.text)
        article_date.append(article.publish_date)
        article_url.append(article.url)
        article_alignment.append(lean)
        article_brand.append(brand)


    article_dict = {
        'Title': article_title,
        'URL': article_url,
        'Text': article_text,
        'Date': article_date,
        "Alignment": lean
    }

    source_df = pd.DataFrame(article_dict)
    return source_df
    

cnn_df = recover_articles(cnn_paper, "left")
fox_df = recover_articles(fox, "right")
bloomberg_df = recover_articles(bloomberg, "left-center")
breitbart_df = recover_articles(breitbart, "right")
cnbc_df = recover_articles(cnbc, "center")
forbes_df = recover_articles(forbes, "center")
huffpo_df = recover_articles(huffpo, "left")
nypost_df = recover_articles(nypost, "right-center")

news_df = pd.concat([cnn_df, fox_df, bloomberg_df, breitbart_df, cnbc_df, forbes_df, huffpo_df, nypost_df])
news_df.to_csv('all_articles.csv', index=False, encoding='utf-8')


