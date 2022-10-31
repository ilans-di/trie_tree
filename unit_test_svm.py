import argparse
import hashlib
import glob
import numpy as np
import pickle
import os
import pandas as pd
from syrah import File

def load_syrah(syrah_path):
    with File(syrah_path, mode='r') as syr:
        items = [syr.get_item(i) for i in range(syr.num_items())]
    return items
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("syrah_path", help="path of where the syrah files are located - expected a folder with"
                                           " train and test folders with syrah files")
    parser.add_argument("parquet_path", help="path of where the parquet files are located - expected a folder with"
                                           " train_data, validation_data and test_data folders with parquet files")
    args = parser.parse_args()
    path_to_ngrams = "/home/ilan/Data/small-vecs/ngrams_with_indices_after_svm"
    path_to_parquet_data = "/home/ilan/Data/small-vecs"
    ngrams = pd.read_parquet(path_to_ngrams)
    ngrams_to_ind = {ngram:ind for ngram,ind in zip(ngrams["ngram"].to_list(),ngrams["index"].to_list())}
    ind_to_ngrams = {ind: ngram for ngram, ind in zip(ngrams["ngram"].to_list(), ngrams["index"].to_list())}
    test_files = glob.glob(os.path.join(path_to_parquet_data,"test/*"))

    pkl_dict = os.path.join(args.syrah_path,"hash2url.pkl")
    if not os.path.exists(pkl_dict):
        all_urls = pd.read_parquet(os.path.join(args.parquet_path, "train_data"))['normalized_url'].to_list()
        all_urls += pd.read_parquet(os.path.join(args.parquet_path, "validation_data"))['normalized_url'].to_list()
        all_urls +=pd.read_parquet(os.path.join(args.parquet_path, "test_data"))['normalized_url'].to_list()
        url2hash = {key:hashlib.sha256(key.encode('utf-8')).hexdigest() for key in all_urls}
        hash2url = {url2hash[key]:key for key in url2hash}
        with open(pkl_dict, "wb") as f_p:
            pickle.dump(hash2url,f_p)
    else:
        with open(pkl_dict,"rb") as f_p:
            hash2url = pickle.load(f_p)
    res = load_syrah(test_files[0])
    this_data_features = {}
    this_data_labels = {}
    for item in res:
        this_data_features[hash2url[item["hash"][0]]] = np.unpackbits(item['features']).astype(bool)
        this_data_labels[hash2url[item["hash"][0]]] = item["label"][0]
        if len(this_data_labels)>=200000:break
    for url in this_data_features:
        features = this_data_features[url]
        label = this_data_labels[url]
        features_chosen = np.where(features)[0]
        ngrams_found = [ind_to_ngrams[ind] for ind in features_chosen]
        all_ngrams = [ngram for ngram in ngrams_to_ind if ngram in url]
        if set(all_ngrams)==set(ngrams_found):
            continue
        else:
            print(url)
            print(all_ngrams)
            print(ngrams_found)
    print("over")