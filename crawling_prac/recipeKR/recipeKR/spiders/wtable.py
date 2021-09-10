import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule


class WtableSpider(CrawlSpider):
    name = 'wtable'
    allowed_domains = ['wtable.co.kr']
    start_urls = ['http://wtable.co.kr/recipes']
    # ", ' >> \", \'으로 조정 필요(DB 입력시 오류 발생)
    rules = (
        Rule(LinkExtractor(allow=r'Items/'), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        item = {}
        #item['domain_id'] = response.xpath('//input[@id="sid"]/@value').get()
        #item['name'] = response.xpath('//div[@id="name"]').get()
        #item['description'] = response.xpath('//div[@id="description"]').get()
        # return item
