local_dev = False

class Env:
    def __init__(self):
        if local_dev:
            self.data_dir = "./data"
        else:
            # need to update this path everytime I login to Cyverse
            self.data_dir = "../data/iplant/home/alexanderecooper/analyses/CSC583_FinalProject"

        self.index_path = f"./index"

env = Env()