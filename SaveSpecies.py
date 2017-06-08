from tqdm import trange
import xml.etree.ElementTree as ET
import os.path


#This is the comment made by Benjamin

path = 'Plants/train_xml/'
highest_id = 36310
natural_plant_species = []
sheet_plant_species = []
natural_file = open('natural_train_species.txt', 'w')
sheet_file = open('sheet_train_species.txt', 'w')
for i in trange(highest_id+1, leave=False):
    this_path = path+str(i)+'.xml'
    if os.path.isfile(this_path):
        tree = ET.parse(this_path)
        root = tree.getroot()
        spec = root.find('ClassId').text
        type = root.find('Type').text
        if type == 'NaturalBackground':
            natural_plant_species.append(spec)
            natural_file.write('%s\n' % spec)
            natural_file.write('%d\n' % i)
        else:
            sheet_plant_species.append(spec)
            sheet_file.write('%s\n' % spec)
            sheet_file.write('%d\n' % i)

natural_file.close()
sheet_file.close()

print(len(natural_plant_species))
print(len(sheet_plant_species))