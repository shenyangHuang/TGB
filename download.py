import os
import os.path as osp
import random
import shutil
import sys
import zipfile
from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from clint.textui import progress

# folder name where all files are downloaded
data_root = "TG_network_datasets"
# zenodo.org id's for the datasets
zen_id = 7213796
zend_id_all = 7008205


class DataSetName(Enum):
    """
    Enums for all the datasets
    """
    CanParl = "CanParl"
    Contacts = "Contacts"
    Enron = "enron"
    Flights = "Flights"
    Lastfm = "lastfm"
    Mooc = "mooc"
    Reddit = "reddit"
    SocialEvo = "SocialEvo"
    UCI = "uci"
    UNtrade = "UNtrade"
    UNvote = "UNvote"
    USLegis = "USLegis"
    Wikipedia = "wikipedia"


# mapping of dataset names to enums.
check_dict = {
    "canparl": DataSetName.CanParl,
    "contacts": DataSetName.Contacts,
    "enron": DataSetName.Enron,
    "flights": DataSetName.Flights,
    "lastfm": DataSetName.Lastfm,
    "mooc": DataSetName.Mooc,
    "reddit": DataSetName.Reddit,
    "socialEvo": DataSetName.SocialEvo,
    "UCI": DataSetName.UCI,
    "un_trade": DataSetName.UNtrade,
    "un_vote": DataSetName.UNvote,
    "us_Legis": DataSetName.USLegis,
    "wikipedia": DataSetName.Wikipedia,
}
# dictionary for all data_sets and files associated with them
sub_dict = {
    "CanParl": ["CanParl.csv", "ml_CanParl.csv", "ml_CanParl.npy", "ml_CanParl_node.npy"],
    "Contacts": ["Contacts.csv", "ml_Contacts.csv", "ml_Contacts.npy", "ml_Contacts_node.npy"],
    "enron": ["ml_enron.csv", "ml_enron.npy", "ml_enron_node.npy"],
    "Flights": ["Flights.csv", "ml_Flights.csv", "ml_Flights.npy", "ml_Flights_node.npy"],
    "lastfm": ["lastfm.csv", "ml_lastfm.csv", "ml_lastfm.npy", "ml_lastfm_node.npy"],
    "mooc": ["ml_mooc.csv", "ml_mooc.npy", "ml_mooc_node.npy", "mooc.csv"],
    "reddit": ["ml_reddit.csv", "ml_reddit.npy", "ml_reddit_node.npy", "reddit.csv"],
    "SocialEvo": ["ml_SocialEvo.csv", "ml_SocialEvo.npy", "ml_SocialEvo_node.npy"],
    "uci": ["ml_uci.csv", "ml_uci.npy", "ml_uci_node.npy"],
    "UNtrade": ["ml_UNtrade.csv", "ml_UNtrade.npy", "ml_UNtrade_node.npy", "UNtrade.csv"],
    "UNvote": ["ml_UNvote.csv", "ml_UNvote.npy", "ml_UNvote_node.npy", "UNvote.csv"],
    "USLegis": ["ml_USLegis.csv", "ml_USLegis.npy", "ml_USLegis_node.npy", "USLegis.csv"],
    "wikipedia": ["ml_wikipedia.csv", "ml_wikipedia.npy", "ml_wikipedia_node.npy", "wikipedia.csv"]
}


class BColors:
    """
    A class to change the colors of the strings.
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'




class Data:
    """
    Data class for processing
    """

    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)


def download_all():
    """
    downloads all data_sets that have not been downloaded yet
    """

    print("verifying then downloading missing files")
    input_list = dataset_names_list()
    for n in input_list:
        _ = TemporalDataSets(data_name=n, skip_download_prompt=True)
    print("missing file download complete")

def force_download_all():
    """
    removes all data set files and redownloads all data_sets that have not been downloaded yet
    """

    print(
        BColors.WARNING + "attempting download all data, will remove ALL files and download ALL possible files" + BColors.ENDC)
    print("To download missing files only, use 'download_all'")
    inp = input('Confirm redownload? (y/N)\n').lower()
    if 'y' == inp:
        try:
            shutil.rmtree(f"./{data_root}")
        except:
            pass

        download_all()
    else:
        print("download cancelled")






def unzip_delete():
    """
    unzips zenodo files and deletes unnecessary files
    """
    os.remove("./md5sums.txt") if os.path.exists("./md5sums.txt") else None

    for filename in Path("../..").glob("*.tmp"):
        filename.unlink()

    if not os.path.exists("./TG_network_datasets.zip"):
        print(f"{BColors.FAIL}DOWNLOAD FAILED{BColors.ENDC}, TG_network_datasets not found")
        return

    with zipfile.ZipFile("TG_network_datasets.zip", 'r') as zip_ref:
        zip_ref.extractall()
    try:
        os.remove("TG_network_datasets.zip")
    except OSError:
        pass
    dirpath = Path('__MACOSX')
    if dirpath.exists():
        shutil.rmtree(dirpath)
    try:
        os.remove("md5sums.txt")
    except OSError:
        pass


def print_dataset_names():
    """
    print name of all datasets
    """
    print("The following is the list of possible dataset names")
    for name in check_dict.keys():
        print(name)


def dataset_names_list():
    """
    returns list of all datasets
    """
    return check_dict.keys()


class TemporalDataSets(object):
    """
    A class used to create data set objects
    """

    def __init__(self, data_name: str = None, data_set_statistics: bool = True, skip_download_prompt: bool = False):
        """
        Parameters
        ----------
        data_name : str
            The name of the data set, to see all possible use::
            >>> print(dgb.list_all())
        data_set_statistics : bool,optional
            False to suppress all data set statistics prints
        skip_download_prompt : bool,optional
            skip download prompt if data not found and move straight to downloading
        """

        self.data_str = data_name
        self.base_directory = f"TG_network_datasets/{self.data_str}"
        self.data_root = "TG_network_datasets"
        self.exception_msg_process = BColors.FAIL + "please run process() method before retrieving training data" + BColors.ENDC
        self.data_set_statistics = data_set_statistics
        self.url = f"https://zenodo.org/record/{zen_id}"
        self.mask = None
        self.train = None
        self.test = None
        self.val = None
        self.skip_download_prompt = skip_download_prompt

        if data_name not in check_dict.keys():

            sys.stdout.write(f" input for TemporalDataSets: '{str(data_name)}' not found, \n inputs must be "
                             f"from the following list: \n")
            for key in check_dict.keys():
                sys.stdout.write(f"   {key}\n")

            inp = input('Exit program ? this is recommended action (y/N)').lower()
            if inp == "y":
                exit()
            else:
                sys.stdout.write(BColors.WARNING + "program will continue but program is unsafe \n" + BColors.ENDC)

        else:
            self.data_list = [check_dict.get(data_name)]  # original name
            self.url += f"/files/{self.data_list[0].value}.zip?download=1"
            self.path_download = f"./{self.data_list[0].value}.zip"

        self.check_downloaded()

    def delete_single(self):
        """
        removes unnecessary files after download
        """
        try:
            os.remove(f"{self.data_list[0].value}.zip")
        except OSError:
            pass
        dirpath = Path('__MACOSX')
        if dirpath.exists():
            shutil.rmtree(dirpath)
        try:
            os.remove("md5sums.txt")
        except OSError:
            pass

    def download_file(self):
        """
        downloads a complete dataset
        """
        if not self.skip_download_prompt:
            print("Data missing, download recommended!")
        inp = "y"
        if not self.skip_download_prompt:
            inp = input('Will you download the dataset(s) now? (y/N)\n').lower()
        if inp == 'y':
            print(f"{BColors.WARNING}Download started, this might take a while . . . {BColors.ENDC}")
            print(f"Dataset title: {self.data_list[0].value}")
            r = requests.get(self.url, stream=True)
            with open(self.path_download, 'wb') as f:
                total_length = int(r.headers.get('content-length'))
                for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length / 1024) + 1):
                    if chunk:
                        f.write(chunk)
                        f.flush()

            os.makedirs(f"./{self.data_root}", exist_ok=True)

            try:
                shutil.rmtree(f"./{self.data_root}/{self.data_list[0].value}")
            except:
                pass

            with zipfile.ZipFile(self.path_download, 'r') as zip_ref:
                zip_ref.extractall(f"./{self.data_root}")

            self.delete_single()
            print(f"{BColors.OKGREEN}Download completed {BColors.ENDC}")

        else:
            raise Exception(
                BColors.FAIL + "Data not found error, download " + self.data_str + " to continue" + BColors.ENDC)

    def check_downloaded(self):
        """
        check if file is properly downloaded
        """
        if not osp.isdir(f"./{self.data_root}"):
            print(f"{BColors.FAIL}folder {self.data_root} not found/ Data not found{BColors.ENDC}")
            self.download_file()
            return
        list_data_not_found = []

        for data_set_name in self.data_list:
            data_found = True
            for file_name in sub_dict[str(data_set_name.value)]:
                path = f"./{self.data_root}/{str(data_set_name.value)}/{file_name}"
                if not Path(path).exists():
                    data_found = False
            if not data_found:
                list_data_not_found.append(data_set_name.value)
        if not list_data_not_found:
            if self.data_set_statistics:
                print(f"All data found for {BColors.OKGREEN}{self.data_str}{BColors.ENDC}")
        else:
            sys.stdout.write("The following datasets not found: ")
            for data_set_name in list_data_not_found:
                sys.stdout.write(f"{data_set_name} ")
            sys.stdout.write("\n")
            self.download_file()

    @property
    def train_data(self):
        if not self.train:
            raise Exception(self.exception_msg_process)
        return self.train

    @property
    def test_data(self):
        if not self.test:
            raise Exception(self.exception_msg_process)
        return self.test

    @property
    def val_data(self):
        if not self.val:
            raise Exception(self.exception_msg_process)
        return self.val

    def process(self):
        """
        processes the data, and returns train, val, test sets.
        """
        # split_masks = {
        #     'train': train_mask,
        #     'val': val_mask,
        #     'nn_val': new_node_val_mask,
        #     'test': new_node_val_mask,
        #     'nn_test': new_node_test_mask,
        # }
        split_masks = self.generate_split_masks(self.data_str)

        # data_splits = {
        #     'node_feats': node_features,
        #     'eddge_feats': edge_features,
        #     'full_data': full_data,
        #     'train_data': train_data,
        #     'val_data': val_data,
        #     'test_data': test_data,
        #     'new_node_val_data': new_node_val_data,
        #     'new_node_test_data': new_node_test_data
        # }
        data_splits = self.get_data_link_pred_from_indices(self.data_str, split_masks=split_masks)

        self.train = data_splits['train_data']
        self.test = data_splits['test_data']
        self.val = data_splits['val_data']
        return {"train": self.train, "test": self.test, "validation": self.val}

    def generate_split_masks(self, dataset_name, val_ratio=0.15, test_ratio=0.15,
                             different_new_nodes_between_val_and_test=False,
                             rnd_seed=2020, save_indices=False):
        """
        only generates the indices of the data in train, validation, and test split
        """
        random.seed(rnd_seed)

        ### Load data and train val test split
        graph_df = pd.read_csv('./{}/ml_{}.csv'.format(self.base_directory, dataset_name))

        val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))
        sources = graph_df.u.values
        destinations = graph_df.i.values
        edge_idxs = graph_df.idx.values
        labels = graph_df.label.values
        timestamps = graph_df.ts.values

        node_set = set(sources) | set(destinations)
        n_total_unique_nodes = len(node_set)

        # Compute nodes which appear at test time
        test_node_set = set(sources[timestamps > val_time]).union(
            set(destinations[timestamps > val_time]))
        # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
        # their edges from training
        new_test_node_set = set(random.sample(list(test_node_set), int(0.1 * n_total_unique_nodes)))

        # Mask saying for each source and destination whether they are new test nodes
        new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
        new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

        # Mask which is true for edges with both destination and source not being new test nodes (because
        # we want to remove all edges involving any new test node)
        observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

        # For train we keep edges happening before the validation time which do not involve any new node
        # used for inductiveness
        train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)

        train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                          edge_idxs[train_mask], labels[train_mask])

        # define the new nodes sets for testing inductiveness of the model
        train_node_set = set(train_data.sources).union(train_data.destinations)
        assert len(train_node_set & new_test_node_set) == 0
        new_node_set = node_set - train_node_set

        val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
        test_mask = timestamps > test_time

        if different_new_nodes_between_val_and_test:
            n_new_nodes = len(new_test_node_set) // 2
            val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
            test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

            edge_contains_new_val_node_mask = np.array(
                [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
            edge_contains_new_test_node_mask = np.array(
                [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
            new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
            new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)


        else:
            edge_contains_new_node_mask = np.array(
                [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
            new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
            new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

        if save_indices:
            print("INFO: Saving index files for {}...".format(dataset_name))
            # split_masks
            np.save('./{}/split_index/ml_{}_val_index.npy'.format(self.base_directory, dataset_name), train_mask)
            np.save('./{}/split_index/ml_{}_test_index.npy'.format(self.base_directory, dataset_name), train_mask)
            np.save('./{}/split_index/ml_{}_new_node_val_index.npy'.format(self.base_directory, dataset_name),
                    train_mask)
            np.save('./{}/split_index/ml_{}_new_node_test_index.npy'.format(self.base_directory, dataset_name),
                    train_mask)

        split_masks = {
            'train': train_mask,
            'val': val_mask,
            'nn_val': new_node_val_mask,
            'test': new_node_val_mask,
            'nn_test': new_node_test_mask,
        }

        return split_masks

    def get_data_link_pred_from_indices(self, dataset_name, split_masks=None, randomize_fatures=False):
        """
        Load data based on the indices read from file
        """
        ### Load data and train val test split
        graph_df = pd.read_csv('./{}/ml_{}.csv'.format(self.base_directory, dataset_name))
        edge_features = np.load('./{}/ml_{}.npy'.format(self.base_directory, dataset_name))
        node_features = np.load('./{}/ml_{}_node.npy'.format(self.base_directory, dataset_name))

        # additional for CAW data specifically
        if dataset_name in ['enron', 'socialevolve', 'uci']:
            node_zero_padding = np.zeros((node_features.shape[0], 172 - node_features.shape[1]))
            node_features = np.concatenate([node_features, node_zero_padding], axis=1)
            edge_zero_padding = np.zeros((edge_features.shape[0], 172 - edge_features.shape[1]))
            edge_features = np.concatenate([edge_features, edge_zero_padding], axis=1)

        if (split_masks is not None):
            train_mask = split_masks['train']
            val_mask = split_masks['val']
            new_node_val_mask = split_masks['nn_val']
            test_mask = split_masks['test']
            new_node_test_mask = split_masks['nn_test']
        else:
            print("INFO: Get split indices from the files...")
            # read index of the data splits
            train_mask = np.load('./{}/split_index/ml_{}_train_index.npy'.format(self.base_directory, self.data_str))
            val_mask = np.load('./{}/split_index/ml_{}_val_index.npy'.format(self.base_directory, self.data_str))
            test_mask = np.load('./{}/split_index/ml_{}_test_index.npy'.format(self.base_directory, self.data_str))
            new_node_val_mask = np.load(
                './{}/split_index/ml_{}_new_node_val_index.npy'.format(self.base_directory, self.data_str))
            new_node_test_mask = np.load(
                './{}/split_index/ml_{}_new_node_test_index.npy'.format(self.base_directory, self.data_str))

        if randomize_fatures:
            node_features = np.random.rand(node_features.shape[0], node_features.shape[1])

        sources = graph_df.u.values
        destinations = graph_df.i.values
        edge_idxs = graph_df.idx.values
        labels = graph_df.label.values
        timestamps = graph_df.ts.values

        full_data = Data(sources, destinations, timestamps, edge_idxs, labels)
        train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                          edge_idxs[train_mask], labels[train_mask])
        val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                        edge_idxs[val_mask], labels[val_mask])
        test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                         edge_idxs[test_mask], labels[test_mask])
        new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                                 timestamps[new_node_val_mask],
                                 edge_idxs[new_node_val_mask], labels[new_node_val_mask])
        new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                                  timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                                  labels[new_node_test_mask])

        if self.data_set_statistics:
            print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                                         full_data.n_unique_nodes))
            print("The training dataset has {} interactions, involving {} different nodes".format(
                train_data.n_interactions, train_data.n_unique_nodes))
            print("The validation dataset has {} interactions, involving {} different nodes".format(
                val_data.n_interactions, val_data.n_unique_nodes))
            print("The test dataset has {} interactions, involving {} different nodes".format(
                test_data.n_interactions, test_data.n_unique_nodes))
            print("The new node validation dataset has {} interactions, involving {} different nodes".format(
                new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
            print("The new node test dataset has {} interactions, involving {} different nodes".format(
                new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))

        data_splits = {
            'node_feats': node_features,
            'edge_feats': edge_features,
            'full_data': full_data,
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'new_node_val_data': new_node_val_data,
            'new_node_test_data': new_node_test_data
        }
        return data_splits