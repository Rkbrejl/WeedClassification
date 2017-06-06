from tqdm import trange
import xml.etree.ElementTree as ET
import os.path


#This is the comment made by Benjamin

path = 'Plants/train_xml/'
highest_id = 36310
plant_species = []
my_file = open('train_species.txt', 'w')
for i in trange(highest_id+1, leave=False):
    this_path = path+str(i)+'.xml'
    if os.path.isfile(this_path):
        tree = ET.parse(this_path)
        root = tree.getroot()
        spec = root.find('ClassId').text
        plant_species.append(spec)
        my_file.write('%s\n' % spec)

my_file.close()

print(len(plant_species))