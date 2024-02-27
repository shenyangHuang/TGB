import csv

def load_index(input_path):
    index, rev_index = {}, {}
    with open(input_path) as f:
        for i, line in enumerate(f.readlines()):        # relaions.dict和entities.dict中的id都是按顺序排列的
            rel, id = line.strip().split("\t")
            index[rel] = id
            rev_index[id] = rel
    return index, rev_index


def load_tab_list(input_path):
    with open(input_path) as f:
        for i, line in enumerate(f.readlines()): 
            u,v,e,t, = line.strip().split("\t")

        

def main():
    """
    concatenate and merge the edgelists into one 
    change tab to ,
    """
    fname = "train.txt"




if __name__ == "__main__":
    main()

