import optparse
from json import JSONEncoder


result = []
for i in range(900, 1501):
    item = {'filename': str(i) + ".png"}
    figures = []
    for j in range(1, 11):
        figure = {}
        figure['height'] = 0
        figure['label'] = j
        figure['left'] = 0
        figure['top'] = 0
        figure['width'] = 0
        figures.append(figure)
    item['boxes'] = figures
    result.append(item)

fout = open('retrain' + ".json",'w')
fout.write(JSONEncoder(indent = True).encode(result))
fout.close()