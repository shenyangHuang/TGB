import csv


with open('ml_uci.csv', 'r', newline='\n') as infile, open('tgbl-uci_edgelist.csv', 'w', newline='\n') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    for row in reader:
        writer.writerow(row[1:])