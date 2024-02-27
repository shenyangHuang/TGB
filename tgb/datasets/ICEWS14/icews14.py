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
    rows = []
    with open(input_path) as f:
        for i, line in enumerate(f.readlines()): 
            head,relation,tail,t, = line.strip().split("\t")
            rows.append([t,head,tail,relation])
    return rows

        
def write2csv(rows, output_path):
    with open(output_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "head", "tail", "relation_type"])
        writer.writerows(rows)


def main():
    """
    concatenate and merge the edgelists into one 
    change tab to ,
    """
    train_name = "train.txt"
    train_rows = load_tab_list(train_name)

    val_name = "valid.txt"
    val_rows = load_tab_list(val_name)

    test_name = "test.txt"
    test_rows = load_tab_list(test_name)

    all_rows = train_rows + val_rows + test_rows
    output_path = "icews14.csv"
    write2csv(all_rows, output_path)







if __name__ == "__main__":
    main()

