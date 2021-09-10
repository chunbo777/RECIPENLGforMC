# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

# Extracted data => Temporary containers (items) => Storing in database


import scrapy


class RecipekrItem(scrapy.Item):
    # define the fields for your item here like:
    # pass
    # title = scrapy.Field()
    # author = scrapy.Field()
    # tag = scrapy.Field()

    title = scrapy.Field()
    ingredients = scrapy.Field()
    directions = scrapy.Field()
    link = scrapy.Field()

