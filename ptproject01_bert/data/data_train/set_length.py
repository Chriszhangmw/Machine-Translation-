import matplotlib.pyplot as plt

datapath = './train.csv'

with open(datapath,'r',encoding='utf-8') as f:
    data = f.readlines()
f.close()

lensentence = {}

for line in data:
    line = line.strip()
    line = line.split("fengefu")
    content = line[0]
    lens1 = len(content)
    title = line[1]
    lens2 = len(title)

    lens = lens1 + lens2
    if lens not in lensentence.keys():
        lensentence[lens] = 1
    else:
        lensentence[lens] += 1

plt.bar(lensentence.keys(),lensentence.values())
plt.show()

