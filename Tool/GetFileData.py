import json

def GetJson():
        with open('D:/上课材料/高级软件工程/code/app/imagenet-simple-labels.json', 'r', encoding='utf-8') as f:
            return json.load(f)
