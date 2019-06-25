import json


class LoadData(object):

    def __init__(self):
        pass

    def parse_lables(self, path):
        with open(path) as json_file:
            data = json.load(json_file)
            for d in data:
                pass
