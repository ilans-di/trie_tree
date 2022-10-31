import numpy as np
from syrah import File
file_path = r"/home/algo/Downloads"
with File(file_path, mode='r') as syr:
    for i in range(syr.num_items()):
        # item is a dictionary with keys 'label' and 'features'
        item = syr.get_item(i)