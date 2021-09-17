import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
import traceback
import json
from recipeKR.items import RecipekrItem
from bs4 import BeautifulSoup


class LampcookSpider(CrawlSpider):
    name = 'lampcook_north'
    allowed_domains = ['lampcook.com']
    # start_urls = [f'http://www.lampcook.com/food/food_local_list.php?pagenum={i}' for i in range(1,326)]
    start_urls = [f'http://www.lampcook.com/food/food_north_list.php?pagenum={i}' for i in range(1, 2833)]
    
    rules = (
        Rule(LinkExtractor(allow=r"food/food_north_view"), callback='parse_item', follow=True , process_links="process_link", process_request="proreq"),
    )
    #allow=r"food/food_north_view"
##div_main_content > div.item_box_data2 > ul:nth-child(1) > li.item_box_thum > a
#<a href="/food/food_north_view.php?idx_no=2272" onclick="openPage(2272); return false;" title="뱀장어련뿌리호두볶음 레시피 바로가기 >"><img src="/wi_files/food_north_img/2272.jpg" alt="뱀장어련뿌리호두볶음" class="txt_vt"></a>
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
            ingredients1=[i.strip() for i in soup.select_one("#div_main_content > div:nth-child(5) > div.padd20").text.split(",")] 
            ingredients2=[i.strip() for i in soup.select_one("#div_main_content > div:nth-child(6) > div.padd20").text.split(",")] 
            ingredients= ingredients1+ingredients2
            
            directions = [soup.select_one(f"#div_main_content > div:nth-child({i+7}) > div.padd20").text.strip() for i in range(len(soup.select("div.padd20"))-5)]
            
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