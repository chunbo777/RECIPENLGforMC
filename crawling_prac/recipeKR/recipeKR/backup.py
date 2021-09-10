import scrapy

from ..items import RecipekrItem

class recipeSpider(scrapy.Spider):
    name = 'quotes'
    # start_urls = ['https://quotes.toscrape.com/']
    page_number = 2
    start_urls = ['https://quotes.toscrape.com/page/1/']# with pagination

    def parse(self, response):
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
        print(next_page)
        if recipeSpider.page_number < 11 :
            recipeSpider.page_number += 1
            yield response.follow(next_page, callback= self.parse)
