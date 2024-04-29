r"""
How to use
python tkgl_wikidata.py --chunk 0 --num_chunks 25
# python tkgl_wikidata.py --chunk 1 --num_chunks 25
"""

from qwikidata.entity import WikidataItem
from qwikidata.json_dump import WikidataJsonDump
from qwikidata.datavalue import get_datavalue_from_snak_dict, WikibaseEntityId
from tqdm import tqdm
from collections import defaultdict
import os.path as osp
import os
import pickle
import argparse
import numpy as np
import csv

def timeEdgeWrite2csv(outname, out_dict):
    with open(outname, 'w') as f:
        writer = csv.writer(f, delimiter =',')
        writer.writerow(['timestamp', 'head', 'tail', 'relation_type', 'time_rel_type'])
        for edge in out_dict.keys():
            ts = edge[0]
            src = edge[1]
            dst = edge[2]
            rel_type = edge[3]
            time_rel_type = edge[4]
            row = [ts, src, dst, rel_type, time_rel_type]
            writer.writerow(row)


def EdgeWrite2csv(outname, out_dict):
    with open(outname, 'w') as f:
        writer = csv.writer(f, delimiter =',')
        writer.writerow(['head', 'tail', 'relation_type'])
        for edge in out_dict.keys():
            src = edge[0]
            dst = edge[1]
            rel_type = edge[2]
            row = [src, dst, rel_type]
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--split', type=str, default = 'train',
    #                     help='an integer for the accumulator')
    parser.add_argument('--chunk', type=int, default = 0,
                        help='an integer for the accumulator')
    parser.add_argument('--num_chunks', type=int, default = 10,
                        help='an integer for the accumulator')

    args = parser.parse_args()
    print(args)
    assert args.chunk < args.num_chunks

    # # create an instance of WikidataJsonDump
    # if args.split == 'train':
    #     wjd_dump_path_original = "wikidata-20210517-all.json.gz"
    # elif args.split == 'val':
    #     wjd_dump_path_original = "wikidata-20210607-all.json.gz"
    # elif args.split == 'test':
    #     wjd_dump_path_original = 'wikidata-20210628-all.json.gz'
    # else:
    #     raise ValueError('Unknown split')

    #* download here
    #? https://dumps.wikimedia.org/wikidatawiki/entities/

    wjd_dump_path_original = "wikidata-20240220-all.json.gz" #"latest-all_03_Apr_2024_12_49.json" 

    wjd_dump_path = osp.join('dump', wjd_dump_path_original)
    wjd = WikidataJsonDump(wjd_dump_path)

    print(wjd_dump_path)

    

    """
    # head = entity_dict['id']
    # type = entity_dict['type']
    # labels = entity_dict['labels']
    # descriptions = entity_dict['descriptions']
    # aliases = entity_dict['aliases']
    # if ('claims' in entity_dict):
    #     claims = entity_dict['claims']
    #     for key in claims.keys():
    #         print (key)
    #         print (claims[key])
    # sitelinks = entity_dict['sitelinks']
    """

    time_edge_dict = {} #{()}
    time_rel_dict = {}
    static_edge_dict = {}
    dummy_rel_set = ['P31','P279']  #filter out instance of and subclass of
    time_rel_set = ['P585','P580', 'P582', 'P577', 'P574']  #point in time, start time, end time, publication date,year of publication of scientific name for taxon

    num_totals = 100000000 #4000000 #10000000 #110000000

    tmp = np.linspace(0, num_totals, args.num_chunks + 1).astype(np.int64)
    start_idx = tmp[args.chunk]
    end_idx = tmp[args.chunk + 1]
    print('Start: ', start_idx)
    print('End: ', end_idx)


    #? output format is (timestamp, head, tail, relation_type, time_rel_type)
    for i, entity_dict in enumerate(tqdm(wjd, total=(end_idx))):
        #! entity_dict keys(['type', 'id', 'labels', 'descriptions', 'aliases', 'claims', 'sitelinks', 'pageid', 'ns', 'title', 'lastrevid', 'modified'])
        if i > end_idx:
            break

        if not (start_idx <= i and i < end_idx):
            continue

        head = entity_dict['id']

        # head needs to start from 'Q'
        if head[0] == 'Q':
            head_id = head
            if 'claims' in entity_dict:
                claim_dict = entity_dict['claims']
                rel_list = list(claim_dict.keys())
                for rel in rel_list:
                    if (rel in dummy_rel_set):
                        continue
                    tail_list = claim_dict[rel]
                    for tail in tail_list:
                        tail_id = None
                        #* first check if there is a valid tail
                        if (tail['mainsnak']['datatype'] == 'wikibase-item'):
                            if ('rank' in tail) and (tail['rank'] != 'deprecated') and ('datavalue' in tail['mainsnak']):
                                if 'id' in tail['mainsnak']['datavalue']['value']:
                                    tail_id = tail['mainsnak']['datavalue']['value']['id']
                                else: 
                                    tail_id = 'Q' + str(tail['mainsnak']['datavalue']['value']['numeric-id'])

                        #* check if there is a qualifier and if it is a time qualifier
                        if (tail_id is not None):
                            if ("qualifiers" in tail):
                                time_logged = False
                                for q in tail["qualifiers"]:
                                    for item in tail["qualifiers"][q]:
                                        if (item['datatype'] == 'time') and ('datavalue' in item):
                                            timestr = item['datavalue']['value']['time']
                                            time_rel_type = q
                                            if (time_rel_type in time_rel_set):
                                                time_edge_dict[(timestr, head_id, tail_id, rel, time_rel_type)] = 1
                                                time_logged = True
                                            else:
                                                time_logged = False
                                if not time_logged:
                                    static_edge_dict[(head_id, tail_id, rel)] = 1
                            else:
                                static_edge_dict[(head_id, tail_id, rel)] = 1

    #! write edges to file
    print ("there are ", len(time_edge_dict), " temporal edges in the dataset")
    outname = "time_edgelist_" + str(args.chunk) + ".csv" #"tkgl-wikidata_time_edgelist.csv"
    timeEdgeWrite2csv(outname, time_edge_dict)


    print ("there are ", len(static_edge_dict), " static edges in the dataset")
    outname = "static_edgelist_" + str(args.chunk) + ".csv" #"tkgl-wikidata_static_edgelist.csv"
    EdgeWrite2csv(outname, static_edge_dict)
    

                
                    
if __name__ == '__main__':

    main()