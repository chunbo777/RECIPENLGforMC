# "html" submodule, e.g. import webapp.html.html

def getScript(filename):
    with open(f'./{filename}', mode='r', encoding='utf8') as f:
        contents = f.read()
        return contents