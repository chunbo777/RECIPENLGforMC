import scrapy
from scrapy.http import FormRequest
from scrapy.utils.response import open_in_browser
from ..items import RecipekrItem

class recipeSpider(scrapy.Spider):
    name = 'quotes'
    start_urls = ['https://quotes.toscrape.com/login']# with pagination
    page_number = 2
    

    def parse(self, response):
        # pass
        token = response.css('form input::attr(value)').extract_first()
        # print('token : ', token)
        return FormRequest.from_response(response=response, formdata={'csrf_token':token, 'username': '이름', 'password':'비밀번호'}, callback=self.start_scraping)

    def start_scraping(self, response):
        
        open_in_browser(response=response)# 앞선 reponse 객체 정보를 가지고 브라우저를 염(login 상태의 페이지가 열려야 함)
        
        # title = response.css('title').extract()

        # title = response.css('title::text').extract()
        # yield {'titletext':title}

        items = RecipekrItem()
        
        all_div_quotes = response.css('div.quote')

        for quote in all_div_quotes:
            title = quote.css('span.text::text').extract()
            author = quote.css('.author::text').extract()
            tag = quote.css('.tag::text').extract()
            
            items['title'] = title
            items['author'] = author
            items['tag'] = tag
            
            yield items# pipeline 으로 이동

        # next_page = response.css('li.next a::attr(href)').get()
        # if next_page is not None:
        #     yield response.follow(next_page, callback= self.parse)

        next_page = 'https://quotes.toscrape.com/page/'+ str(recipeSpider.page_number) +'/'
        # print(next_page)
        if recipeSpider.page_number < 11 :
            recipeSpider.page_number += 1
            yield response.follow(next_page, callback= self.start_scraping)
