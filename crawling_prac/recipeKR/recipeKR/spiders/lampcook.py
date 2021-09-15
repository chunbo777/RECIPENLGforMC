import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
import traceback
import json
from recipeKR.items import RecipekrItem
from bs4 import BeautifulSoup


class LampcookSpider(CrawlSpider):
    name = 'lampcook'
    allowed_domains = ['www.lampcook.com']
    # start_urls = [f'http://www.lampcook.com/food/food_local_list.php?pagenum={i}' for i in range(1,326)]
    start_urls = [f'http://lampcook.com/food/food_north_list.php?pagenum={i}' for i in range(1, 10)]
    
    rules = (
        Rule(LinkExtractor(allow=r"/food/food_north_view"), callback='parse_item', follow=True , process_links="process_link", process_request="proreq"),
    )

    # rules = (
    #     Rule(LinkExtractor(), callback='parse_page', follow=True, process_links="process_link"),
    # )
    def process_link(self, response):
        return response
    
    def proreq(self, request, response):
        return request

    def parse_item(self, response):
        if "/food/" not in response.request.url:
            return
        try:
            item = RecipekrItem()
            soup = BeautifulSoup(response.text, "html.parser")
            
            
            title = soup.select_one('#div_main_content > ul.glo_ul_content > li:nth-child(2) > div > h1').text.strip()
            ingredients=[i.strip() for i in soup.select_one("#div_main_content > div:nth-child(3) > div.padd20").text.split(",")] 
            directions = [soup.select_one(f"#div_main_content > div:nth-child({i+4}) > div.padd20").text.strip() for i in range(len(soup.select("div.padd20"))-4)]
            
            # directions=soup.select(f"#div_main_content > div:nth-child(4) > div.padd20")
            # #div_main_content > div:nth-child(5) > div.padd20
            # ingredients = [' '.join([j.text.strip() for j in i.contents if j is not None and j.text.strip()!='']).strip() for i in soup.select('ul.lst_ingrd>li')]
            # directions = [i.text.strip() for i in soup.select('section.sec_rcp_step>ol.lst_step>li>p') if i is not None and i.text.strip()!='']
            link = response.url

            item['title'] = title
            item['ingredients'] = ingredients
            item['directions'] = directions
            item['link'] = link
        
            yield item
        except Exception as e:
            traceback.print_exc()
        
        # item = {}
        # #item['domain_id'] = response.xpath('//input[@id="sid"]/@value').get()
        # #item['name'] = response.xpath('//div[@id="name"]').get()
        # #item['description'] = response.xpath('//div[@id="description"]').get()
        # return item
     
     
    #  ef parse_item(self, response):
    #     if '/recipes/' not in response.request.url:
    #         return
    #     try:
    #         item = RecipekrItem()
    #         soup = BeautifulSoup(response.text, "html.parser")

    #         title = soup.select_one('section.sec_info>div.aside>div.top>h1>strong').text.strip()
    #         ingredients = [' '.join([j.text.strip() for j in i.contents if j is not None and j.text.strip()!='']).strip() for i in soup.select('ul.lst_ingrd>li')]
    #         directions = [i.text.strip() for i in soup.select('section.sec_rcp_step>ol.lst_step>li>p') if i is not None and i.text.strip()!='']
    #         link = response.url

    #         item['title'] = title
    #         item['ingredients'] = ingredients
    #         item['directions'] = directions
    #         item['link'] = link

    #         yield item
    #     except Exception as e:
    #         traceback.print_exc()