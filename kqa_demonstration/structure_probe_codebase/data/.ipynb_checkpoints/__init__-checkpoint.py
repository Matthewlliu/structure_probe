from data.kopl_dataset import kopl_data
from data.overnight_dataset import overnight_data
from utils import ensemble_input,levenshtein


def get_data_by_name(dataset_name):
    return {
        'kopl': kopl_data,
        'lambdaDCS': overnight_data,
    }[dataset_name]

def get_Dataset(args):
    data_cls = get_data_by_name(args.logic_forms)
    dataset = data_cls(args)
    return dataset