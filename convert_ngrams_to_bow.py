import json
import os
import urllib
import shutil

import h5py
import psutil
import tqdm
import psutil
import numpy as np
from nltk import ngrams
from syrah import File

from create_test_set_without_common_subdomains import get_clean_url
from file_utils import read_parquet
class N_grams():

    def __init__(self, len_ngrams = 0,path = None):
        if path:
            with open(path, "r") as f_p:
                ngrams_extracted = json.load(f_p)
            self.tlds = {tld[0]:k for k,tld in enumerate(ngrams_extracted["tld"])}
            self.domain = {domain_ngrams:k+len(self.tlds) for k,domain_ngrams in enumerate(ngrams_extracted["domain_names"])}
            self.path = {path_ngrams:k+len(self.tlds) + len(self.domain) for k,path_ngrams in enumerate(ngrams_extracted["pathes"])}
            self.len_ngrams = len_ngrams
    def __len__(self):
        return len(self.tlds) + len(self.domain) + len(self.path)

    def get_ngrams_domain(self,clean_url):
        return {"".join(n_gram) for i in range(2, self.len_ngrams) for n_gram in
         ngrams(clean_url, i + 1) if "".join(n_gram) in self.domain}

    def get_bow_domain(self,clean_url):
        return {self.domain["".join(n_gram)] for i in range(2, self.len_ngrams) for n_gram in
         ngrams(clean_url, i + 1) if "".join(n_gram) in self.domain}

    def get_ngrams_path(self,clean_url):
        return {"".join(n_gram) for i in range(2, self.len_ngrams) for n_gram in
         ngrams(clean_url, i + 1) if "".join(n_gram) in self.path}

    def get_bow_path(self,clean_url):
        return {self.path["".join(n_gram)] for i in range(2, self.len_ngrams) for n_gram in
         ngrams(clean_url, i + 1) if "".join(n_gram) in self.path}


    def write(self,path):
        data = {"tld": [[tld,0]for tld in self.tld], "domain_names": ngrams_list_domain, "pathes": ngrams_list_path}
        with open(path,"w") as f_p:

            json.dump()


# {"tld":tld_list,"domain_names":ngrams_list_domain,"pathes":ngrams_list_path}
if __name__=="__main__":
    ngrams_orig = N_grams(7, "/home/algo/Data/Balanced_02_10_2022_08_19_56/important_ngrams_7.json")
        #{"tld":tld_list,"domain_names":ngrams_list_domain,"pathes":ngrams_list_path}
    path_of_data = "/home/algo/Data/Balanced_debug_02_10_2022_08_19_56/"

    all_dss = read_parquet(path_of_data, split=["train_data","validation_data","test_data"], debug=False,
                           what_to_read=["normalized_url","label"])

    number_of_ds_in_mem = 10000

    for category in all_dss:
        print("processing {}".format(category))
        hdf5_path_orig = os.path.join(path_of_data, "{}.hdf5".format(category))
        hdf5_path_selected = os.path.join(path_of_data, "{}_selected.hdf5".format(category))
        if os.path.exists(hdf5_path_orig):
            os.remove(hdf5_path_orig)
        with h5py.File(hdf5_path_orig, 'a') as hf:
            current_features = np.zeros((number_of_ds_in_mem,len(ngrams_orig)),dtype=bool)
            current_labels = np.zeros(number_of_ds_in_mem,dtype=bool)
            ind_write = 0
            for url,label in tqdm.tqdm(all_dss[category][["normalized_url","label"]].itertuples(index=False),total=len(all_dss[category])):
                url = urllib.parse.unquote(url)
                try:
                    clean_url, tld = get_clean_url(url)
                    if tld:
                        path_to_file = url.replace(clean_url + ".{}".format(tld), "")

                        ind_of_ngrams = {ngrams_orig.tlds[tld]}| (ngrams_orig.get_bow_domain(clean_url))

                    else:
                        continue

                    if path_to_file:
                        ind_of_ngrams |= ngrams_orig.get_bow_path(path_to_file)
                    current_features[ind_write%number_of_ds_in_mem,np.fromiter(ind_of_ngrams,dtype=int)] = True

                    current_labels[ind_write%number_of_ds_in_mem] = label
                    ind_write += 1
                    if ind_write == number_of_ds_in_mem:
                        hf.create_dataset("data", data=current_features, maxshape=(None, len(ngrams_orig)))
                        hf.create_dataset("label",data=current_labels,maxshape=(None,))
                    elif ind_write%number_of_ds_in_mem == 0:
                        hf['data'].resize((hf['data'].shape[0] + current_features.shape[0]), axis=0)
                        hf['data'][-current_features.shape[0]:] = current_features

                        hf['label'].resize((hf['label'].shape[0] + current_labels.shape[0]), axis=0)
                        hf['label'][-current_labels.shape[0]:] = current_labels

                except:
                    pass
            if ind_write % number_of_ds_in_mem > 0:
                current_features = current_features[:ind_write % number_of_ds_in_mem]
                current_labels = current_labels[:ind_write % number_of_ds_in_mem]
                hf['data'].resize((hf['data'].shape[0] + current_features.shape[0]), axis=0)
                hf['data'][-current_features.shape[0]:] = current_features

                hf['label'].resize((hf['label'].shape[0] + current_labels.shape[0]), axis=0)
                hf['label'][-current_labels.shape[0]:] = current_labels










