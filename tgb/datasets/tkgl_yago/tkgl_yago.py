import csv
import datetime
import glob, os



def main():
    train_fname = "train.txt"
    val_fname = "valid.txt"
    test_fname = "test.txt"
    
    train_dict, num_lines = load_csv(train_fname)
    print ("there are ", num_lines, " lines in the train file")
    print ("there are ", len(train_dict), " timestamps in the train file")
    val_dict, num_lines = load_csv(val_fname)
    print ("there are ", num_lines, " lines in the val file")
    print ("there are ", len(val_dict), " timestamps in the val file")
    test_dict, num_lines = load_csv(test_fname)
    print ("there are ", num_lines, " lines in the test file")
    print ("there are ", len(test_dict), " timestamps in the test file")

    train_dict.update(val_dict)
    train_dict.update(test_dict)
    print ("there are ", len(train_dict), " timestamps in the combined file")

    outname = "tkgl-yago_edgelist.csv"
    write_csv(outname, train_dict)


def write_csv(outname, out_dict):
    with open(outname, 'w') as f:
        writer = csv.writer(f, delimiter =',')
        writer.writerow(['timestamp', 'head', 'tail', 'relation_type'])
        for ts in out_dict:
            for edge in out_dict[ts]:
                src = edge[0]
                rel_type = edge[1]
                dst = edge[2]
                row = [ts, src, dst, rel_type]
                writer.writerow(row)


def load_csv(fname):
    out_dict = {}
    num_lines = 0
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter ='\t')
        #! src rel_type dst ts 
        # 10289	9	10290	0	0
        for row in reader: 
            src = int (row[0])
            rel_type = int (row[1])
            dst = int (row[2])
            ts = int (row[3])
            if ts not in out_dict:
                out_dict[ts] = {(src,rel_type,dst):1}
            else:
                out_dict[ts][(src,rel_type,dst)] = 1
            num_lines += 1
    return out_dict, num_lines



if __name__ == "__main__":
    main()