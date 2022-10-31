from collections import defaultdict
import os
import re

from tld import get_tld
from tqdm import tqdm
import numpy as np

from file_utils import read_parquet


def get_clean_url(url):
    try:
        tld = get_tld(url)
        if url.count('/')==2:
            return url[:url.rfind(tld) - 1], tld
        else:
            return url[:url.find(tld+"/") - 1], tld
    except:
        return None, None


if __name__ == "__main__":
    path = "Malicious_domain_detection"
    all_dss_without_test = read_parquet(path, split=["training_split", "validation_split"], debug=True)
    all_urls_normalized = []
    all_labels = []
    for key in all_dss_without_test:
        all_urls_normalized += all_dss_without_test[key]["normalized_url"].to_list()
        all_labels +=all_dss_without_test[key]["label"].to_list()
    clean_urls = dict()
    for i in tqdm(range(len(all_urls_normalized))):
        url = all_urls_normalized[i]
        label = all_labels[i]
        clean_url = get_clean_url(url)
        if clean_url not in clean_urls:
            clean_urls[clean_url] = label
        elif clean_urls[clean_url] == 1-label:
            clean_urls[clean_url] = -1

    test_ds = read_parquet(path, split=["test_split"], debug=True)
    what_indices_to_drop = set()
    for k,datum in enumerate(test_ds["test_split"].iterrows()):
        clean_url = get_clean_url(datum[1]["normalized_url"])
        if clean_url in clean_urls:
            what_indices_to_drop.add(k)

    ds_new = test_ds["test_split"].drop(test_ds["test_split"].index[list(what_indices_to_drop)])
    try:
        os.mkdir(os.path.join(path, 'test_without_overlaps'))
    except:
        pass
    ds_new.to_parquet(os.path.join(path,'test_without_overlaps','f1.parquet'))



