import scrapy
from scrapy.http.request import Request
from ..items import RecipekrItem
import time
from scrapy.utils.response import open_in_browser


class SsgSpider(scrapy.Spider):
    name = 'ssg'
    allowed_domains = ['http://emart.ssg.com/', 'emart.ssg.com']
    recipe_idx = 3840
    handle_httpstatus_list = [401, 404]

    download_delay = 1.5
    # sleeptime = 4.0
    # start_urls = [f'http://emart.ssg.com/recipe/recipe/detail.ssg?recipeId={recipe_idx}']
    def start_requests(self):
        # recipe_idx = 1
        urls = [
            f'http://emart.ssg.com/recipe/recipe/detail.ssg?recipeId={SsgSpider.recipe_idx}',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)


    # def getAuth(self, response):
    #     # open_in_browser(response=response)
    #     SsgSpider.recipe_idx += 1
    #     next_recipe = 'http://emart.ssg.com/recipe/recipe/detail.ssg?recipeId='+ str(SsgSpider.recipe_idx)
    #     yield response.follow(next_recipe, callback= self.parse)

    def parse(self, response):
        if response.status == 200:
            items = RecipekrItem()
            title = response.css('h2.post_subject::text').extract()
            try:
                if len(title)>0:
                    ingredients = response.css('dl.recipe_ingredient a.btn_hash::text').extract()
                    directions = response.css('dl.recipe_step p.dsc::text').extract()
                    link = response.url
                
                    items['title'] = title
                    items['ingredients'] = ingredients
                    items['directions'] = directions
                    items['link'] = link
                
                    # yield items# pipeline 으로 이동
                    if items['ingredients'] and items['directions']:
                        # print(items)
                        yield items

                    # # 레시피 정보가 이어져 있어야 함
                    # SsgSpider.recipe_idx += 1
                    # next_recipe = 'http://emart.ssg.com/recipe/recipe/detail.ssg?recipeId='+ str(SsgSpider.recipe_idx)
                    # yield response.follow(next_recipe, callback= self.parse)
                    # self.logger.info("Parsed: "+response.url)
                else:
                    self.logger.warning("Unable to get recipe from: " + response.url)
            except:
                self.logger.warning("Unable to get recipe from: " + response.url)
            # customizedMeta = response.request.meta
            # customizedMeta['proxy'] = 'http://177.200.206.169:5678'
            # customizedMeta['proxy'] = 'http://52.78.172.171:80'
            # customizedMeta['proxy'] = 'http://172.217.31.174:80'
            SsgSpider.recipe_idx += 1
            next_recipe = 'http://emart.ssg.com/recipe/recipe/detail.ssg?recipeId='+ str(SsgSpider.recipe_idx)
            yield response.follow(next_recipe, callback= self.parse)
            # yield scrapy.Request(next_recipe, callback=self.parse, dont_filter=True, meta=customizedMeta)
        else:
            open_in_browser(response=response)
            self.logger.warning("Unable to get recipe from: " + response.url)
            time.sleep(SsgSpider.sleeptime)
            # SsgSpider.sleeptime += 1.0
            # SsgSpider.recipe_idx += 1
            # customizedMeta = response.request.meta
            # customizedMeta['proxy'] = 'http://172.217.31.174:2404'
            SsgSpider.recipe_idx += 1
            next_recipe = 'http://emart.ssg.com/recipe/recipe/detail.ssg?recipeId='+ str(SsgSpider.recipe_idx)
            yield scrapy.Request(next_recipe, callback=self.parse, dont_filter=True)
            # yield scrapy.Request(next_recipe, callback=self.parse, dont_filter=True, meta=customizedMeta)

            # recaptchaParam = response.css('.g-recaptcha').attrib['data-sitekey']
            # yield scrapy.FormRequest('http://emart.ssg.com/google/reCaptcha/verify.ssg',
            #                         formdata={'recaptchaParam': recaptchaParam,},
            #                         callback=self.getAuth, dont_filter=True)            
        # next_page = response.css('li.next a::attr(href)').get()
        # if next_page is not None:
        #     yield response.follow(next_page, callback= self.parse)

        
        # if SsgSpider.recipe_idx < 11 :
        #     SsgSpider.recipe_idx += 1
        #     next_recipe = 'http://emart.ssg.com/recipe/recipe/detail.ssg?recipeId='+ str(SsgSpider.recipe_idx)
        #     yield response.follow(next_recipe, callback= self.parse)
        #     # yield response.follow(next_recipe, callback= self.parse)
