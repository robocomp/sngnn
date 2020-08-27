import json
import os
import sys
directory_path = ""
if len(sys.argv) == 2:
    print(sys.argv)
    directory_path = sys.argv[1]
else:
    print("Run in this format python usecase/mirroring_data directory_path")

for filename in os.listdir(directory_path):
	save = "mirror_"+filename
	if 'mirror' in filename:
		continue
	# Read JSON data into the datastore variable
	if filename:
	    with open(directory_path+'/'+filename, 'r') as f:
	        datastore = json.load(f)
	        f.close()

	for data in datastore:
		data['command'][2] = -data['command'][2]

		for i in range(len(data['goal'])):
			data['goal'][i]['x'] = -data['goal'][i]['x']

		for i in range(len(data['objects'])):
			data['objects'][i]['a'] = 3.14-data['objects'][i]['a']
			data['objects'][i]['x'] = -data['objects'][i]['x']
			data['objects'][i]['vx'] = -data['objects'][i]['vx']
			data['objects'][i]['va'] = 3.14-data['objects'][i]['va']

		for i in range(len(data['people'])):
			data['people'][i]['a'] = 3.14-data['people'][i]['a']
			data['people'][i]['x'] = -data['people'][i]['x']
			data['people'][i]['vx'] = -data['people'][i]['vx']
			data['people'][i]['va'] = 3.14-data['people'][i]['va']

		for i in range(len(data['walls'])):
			data['walls'][i]['x1'] = -data['walls'][i]['x1']
			data['walls'][i]['x2'] = -data['walls'][i]['x2']

	with open("mirrored_data/"+save, "w") as outfile: 
	    json.dump(datastore, outfile) 
	    outfile.close()
