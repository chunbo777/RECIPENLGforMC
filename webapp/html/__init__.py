# makes "html" a "Python subpackage"

def getContents(filename):
    with open(f'./{filename}', mode='r', encoding='utf8') as f:
        contents = f.read()
        return contents
