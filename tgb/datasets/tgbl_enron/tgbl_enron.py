import csv


with open('ml_enron.csv', 'r', newline='\n') as infile, open('tgbl-enron_edgelist.csv', 'w', newline='\n') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    for row in reader:
        writer.writerow(row[1:])