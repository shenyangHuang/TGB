r"""
How to use
python tkgl_wikidata.py --chunk 0 --num_chunks 10
python tkgl_wikidata.py --chunk 1 --num_chunks 10
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

    wjd_dump_path_original = "wikidata-20240220-all.json.gz"

    wjd_dump_path = osp.join('dump', wjd_dump_path_original)
    wjd = WikidataJsonDump(wjd_dump_path)

    print(wjd_dump_path)
    num_totals = 95000000

    tmp = np.linspace(0, num_totals, args.num_chunks + 1).astype(np.int64)
    start_idx = tmp[args.chunk]
    end_idx = tmp[args.chunk + 1]


    start_idx = 0
    end_idx = 2
    print('Start: ', start_idx)
    print('End: ', end_idx)


    for i, entity_dict in enumerate(tqdm(wjd, total=num_totals)):
        #! entity_dict keys(['type', 'id', 'labels', 'descriptions', 'aliases', 'claims', 'sitelinks', 'pageid', 'ns', 'title', 'lastrevid', 'modified'])

        head = entity_dict['id']
        type = entity_dict['type']
        labels = entity_dict['labels']
        descriptions = entity_dict['descriptions']
        aliases = entity_dict['aliases']
        if ('claims' in entity_dict):
            claims = entity_dict['claims']
            print (claims.keys())
        sitelinks = entity_dict['sitelinks']

        print (head)
        # print (type)  #always item
        print (labels['en'])
        print (descriptions['en'])
        # print (aliases.keys())
        # print (sitelinks)


        if i > end_idx:
            break

        if not (start_idx <= i and i < end_idx):
            continue








    # # triplet
    # hrt_list = []
    # dummy_rel_set = set([])
    
    # num_totals = 10 #95000000

    # dir_name = osp.join('processed/', '-'.join(wjd_dump_path_original.split('-')[:2]))
    # os.makedirs(dir_name, exist_ok=True)

    # tmp = np.linspace(0, num_totals, args.num_chunks + 1).astype(np.int64)
    # print(tmp)
    # start_idx = tmp[args.chunk]
    # end_idx = tmp[args.chunk + 1]

    # print('Start: ', start_idx)
    # print('End: ', end_idx)
    # filename = osp.join(dir_name, f'hrt_list_{args.chunk}_{args.num_chunks}.pkl')
    # print(filename)


    # for i, entity_dict in enumerate(tqdm(wjd, total=num_totals)):

    #     if i > end_idx:
    #         break

    #     if not (start_idx <= i and i < end_idx):
    #         continue

    #     head = entity_dict['id']

    #     # head needs to start from 'Q'
    #     if head[0] == 'Q':
    #         if 'claims' in entity_dict:
    #             claim_dict = entity_dict['claims']
    #             rel_list = list(claim_dict.keys() - dummy_rel_set)
    #             for rel in rel_list:
    #                 tail_list = claim_dict[rel]
    #                 if rel[0] == 'P':
    #                     for tail in tail_list:
    #                         if 'datatype' in tail['mainsnak']:
    #                             if tail['mainsnak']['datatype'] == 'wikibase-item':
    #                                 if 'rank' in tail and tail['rank'] != 'deprecated':
    #                                     if 'datavalue' in tail['mainsnak']:
    #                                         #print(tail['mainsnak']['datavalue']['value']['id'])
    #                                         if 'id' in tail['mainsnak']['datavalue']['value']:
    #                                             tail_id = tail['mainsnak']['datavalue']['value']['id']
    #                                         else: 
    #                                             tail_id = 'Q' + str(tail['mainsnak']['datavalue']['value']['numeric-id'])

    #                                         if tail_id[0] == 'Q':
    #                                             hrt_list.append((head, rel, tail_id))
    #                             else:
    #                                 dummy_rel_set.add(rel)
    #                                 break

    # # print(hrt_list)
    # print(len(hrt_list))
    # print (hrt_list)

    # ## did not ensure head[0] == 'Q', rel[0] == 'P' and tail[0] == 'Q'
    # with open(filename, 'wb') as outfile:
    #     pickle.dump(hrt_list, outfile, pickle.HIGHEST_PROTOCOL)
    

if __name__ == '__main__':

    main()