import re
import sys
import json
from tqdm import tqdm


in_file = sys.argv[1]
out_file = sys.argv[2]


with open(in_file) as in_f, open(out_file, 'w') as out_f:
    print(out_file)
    data = []
    for line in tqdm(in_f.readlines()):
        entry = json.loads(line)
        out_entry = {"instruction": "Generate the key phrases for the following document.",
                     "input": entry['src'],
                     "output": re.sub(' ; ', ', ', entry['tgt'])}
        data.append(out_entry)
    json.dump(data, out_f)
      
