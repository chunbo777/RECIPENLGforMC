import os
from scrapy.cmdline import execute

os.chdir(os.path.dirname(os.path.realpath(__file__)))

try:
    execute(
        # ["scrapy", "crawl", "ssg"]
        # ["scrapy", "crawl", "wtable"]
        # ["scrapy", "crawl", "haemuk"]
        # ["scrapy", "crawl", "ssgrecipe"]
        # ["scrapy", "crawl", "manrecipe1"]
        # ["scrapy", "crawl", "chungjungone"]
        # ["scrapy", "crawl", "chungjungonetable"]
        # ["scrapy", "crawl", "philips"]
        # ["scrapy", "crawl", "cheiljedangrecipe"]
        # ["scrapy", "crawl", "manrecipe"]
        # ["scrapy", "crawl", "quotes"]
        # ["scrapy", "crawl", "lampcook"]
        ["scrapy", "crawl", "example"]
        # ["scrapy", "crawl", "lampcook_fusion"]
        # ["scrapy", "crawl", "lampcook_wellbeing"]
        # ["scrapy", "crawl", "lampcook_real"]
        # ["scrapy", "crawl", "lampcook_north"]
    )
except SystemExit:
    pass