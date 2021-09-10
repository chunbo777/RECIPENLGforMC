import os
from scrapy.cmdline import execute

os.chdir(os.path.dirname(os.path.realpath(__file__)))

try:
    execute(
        # ["scrapy", "crawl", "ssg"]
        # ["scrapy", "crawl", "wtable"]
        # ["scrapy", "crawl", "haemuk"]
        # ["scrapy", "crawl", "ssgrecipe"]
        ["scrapy", "crawl", "manrecipe1"]
        # ["scrapy", "crawl", "manrecipe"]
        # ["scrapy", "crawl", "quotes"]
    )
except SystemExit:
    pass