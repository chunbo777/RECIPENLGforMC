import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
import json
from recipeKR.items import RecipekrItem
from bs4 import BeautifulSoup
import traceback

class HaemukSpider(CrawlSpider):
    name = 'haemuk'
    allowed_domains = ['haemukja.com']

    start_urls = [f'https://haemukja.com/recipes?page={i}' for i in range(449)]

    rules = (
        Rule(LinkExtractor(allow=(r'/recipes/[0-9]+',), deny=(r'/recipes/[0-9]+/')), callback='parse_item', follow=True),
        # Rule(LinkExtractor(allow=(r'/recipes/[0-9]+',), deny=(r'/recipes/[0-9]+/')), callback='parse_item', follow=True, process_links='process_link', process_request='process_req'),
        # Rule(LinkExtractor(), follow=True)# 디버그 용
    )

    # def process_link(self, response):
    #     print(response)#debug
    #     return response
    # def process_req(self, request, response):
    #     print(request)
    #     return request#debug

    def parse_item(self, response):
        if '/recipes/' not in response.request.url:
            return
        try:
            item = RecipekrItem()
            soup = BeautifulSoup(response.text, "html.parser")

            title = soup.select_one('section.sec_info>div.aside>div.top>h1>strong').text.strip()
            ingredients = [' '.join([j.text.strip() for j in i.contents if j is not None and j.text.strip()!='']).strip() for i in soup.select('ul.lst_ingrd>li')]
            directions = [i.text.strip() for i in soup.select('section.sec_rcp_step>ol.lst_step>li>p') if i is not None and i.text.strip()!='']
            link = response.url

            item['title'] = title
            item['ingredients'] = ingredients
            item['directions'] = directions
            item['link'] = link

            yield item
        except Exception as e:
            traceback.print_exc()
