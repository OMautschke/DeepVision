import json

reduced_file = {}

with open('bdd100k_labels_images_val.json') as json_file:  
    data = json.load(json_file)
    for pic in data:
        reduced_file[pic['name']] = {
            'attriutes': pic['attributes'],
            'objects': []     
        }
        for obj in pic['labels']:
            if obj['category'] == 'lane' or obj['category'] == 'drivable area':
                continue
            reduced_file[pic['name']]['objects'].append({
                'category': obj['category'],
                'box2d': obj['box2d']
            })


with open('reduced_labels.json', 'w') as outfile:  
    json.dump(reduced_file, outfile)
        