from konlpy.tag import Komoran


text = '''
["올레오 3개", "설탕 3컵", "계란 5개", "밀가루 3 컵.", "베이킹파우더 1/2작은술", "소금 1/2작은술", "코코아 4작은술" ", "우유 1컵", "바닐라 1큰술"] 
'''
ko = Komoran()

morph = ko.morph(text)
print(morph)