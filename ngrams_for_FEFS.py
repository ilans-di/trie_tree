import pickle
from collections import defaultdict, Counter
import urllib
import json
import os
from ngrams_ilan import ngrams_ilan as ngrams_string
import numpy as np

import pandas as pd
import tqdm
import psutil

from create_test_set_without_common_subdomains import get_clean_url


def threshold_counter(counter, threshold):
    ngrams_list_domain = {}
    for key, value in counter.most_common():
        if (value < threshold):
            break
        ngrams_list_domain[key] = value
    return ngrams_list_domain


class Meaninigfull_Ngrams:
    def __init__(self, path_data, len_ngrams):
        self.add_benign = np.array((1, 0), dtype=np.uint32)
        self.add_malicious = np.array((0, 1), dtype=np.uint32)
        self.len_ngrams = len_ngrams
        self.threshold_for_removal = 100000


        self.take_k_largest_chi_square = [1000, 5000, 10000, 20000, 40000, 100000, 200000]
        self.path_data = path_data
        self.ilan_ngrams_time = 0.0
        self.numba_ngrams_time = 0.0

    def load_ds(self,path_data):
        self.ds = pd.read_parquet(path_data, columns=["normalized_url", "label"])
        self.data_len = len(self.ds)
        total_malicious = sum(self.ds["label"])
        total_benign = self.data_len - total_malicious
        self.total_benign_malicious_vec = np.array((total_benign, total_malicious), np.uint32).reshape(1, 2)

    @classmethod
    def threshold_numpy_counter(cls,counter, threshold,return_as_dict=False):
        keys = list(counter)
        values = np.array(list(counter.values()))
        relevant_keys = np.max(values, axis=1) > threshold
        keys = np.array(keys)[relevant_keys].tolist()
        values = values[relevant_keys]
        if not return_as_dict:
            return defaultdict(lambda: np.zeros(2, dtype=np.int32), dict(zip(keys, values)))
        else:
            return dict(zip(keys, values))


    def load_pickle(self, path):
        dir_,dataname = os.path.split(path_data)
        self.pkl_path = os.path.join(dir_, "{}_ngrams_split_{}.pkl".format(dataname, self.len_ngrams))
        if os.path.exists(self.pkl_path):
            with open(self.pkl_path, "rb") as f_p:
                (self.ngrams_dict_path, self.ngrams_dict_domain,self.tld_counter,self.http_or_s, self.data_len,
                 self.total_benign_malicious_vec) = pickle.load(f_p)
            return True
        else:
            self.init()
            self.load_ds(path_data)
            return False

    def init(self):
        self.ngrams_dict_path = defaultdict(lambda: np.zeros(2, dtype=np.uint32))
        self.ngrams_dict_domain = defaultdict(lambda: np.zeros(2, dtype=np.uint32))
        self.tld_counter = defaultdict(lambda: np.zeros(2, dtype=np.uint32))
        self.http_or_s = defaultdict(lambda: np.zeros(2, dtype=np.uint32))

    def clean_memory(self, threshold=2):
        threshold = max(threshold,2)
        print("before domain length = " + str(len(self.ngrams_dict_domain)))
        print("before path length = " + str(len(self.ngrams_dict_path)))
        self.ngrams_dict_domain = self.threshold_numpy_counter(self.ngrams_dict_domain, threshold)
        self.ngrams_dict_path = self.threshold_numpy_counter(self.ngrams_dict_path, threshold)
        print("after domain length = " + str(len(self.ngrams_dict_domain)))
        print("after path length = " + str(len(self.ngrams_dict_path)))

    def run_all(self):
        clean_every = 500000
        urls = self.ds["normalized_url"].to_list()
        labels = np.array(self.ds["label"].to_list(),dtype=bool)

        for ind, item in tqdm.tqdm(self.ds.iterrows(), total=len(self.ds)):
            url = urllib.parse.unquote(item.normalized_url)
            label = item.label
            self.run_single(url, label)
            if ind%clean_every==0 and psutil.virtual_memory().percent > 60:
                self.clean_memory(ind//(self.threshold_for_removal*3))
                print(self.ilan_ngrams_time)
                print(self.numba_ngrams_time)
        print("total number of path ngrams " + str(len(self.ngrams_dict_path)))
        print("total number of domain ngrams " + str(len(self.ngrams_dict_domain)))
        print("total number of tld " + str(len(self.tld_counter)))

        self.ngrams_dict_path = dict(self.ngrams_dict_path)
        self.ngrams_dict_domain = dict(self.ngrams_dict_domain)
        self.tld_counter = dict(self.tld_counter)
        self.http_or_s = dict(self.http_or_s)
        chosen_ngrams = (self.ngrams_dict_path, self.ngrams_dict_domain,self.tld_counter,self.http_or_s, self.data_len,
                         self.total_benign_malicious_vec)
        with open(self.pkl_path, "wb") as f_p:
            pickle.dump(chosen_ngrams,f_p)
        print("save {} as complete file".format(self.pkl_path))


    def run_single(self, url, label):
        add_benign_or_malicious = self.add_malicious if label else self.add_benign
        clean_url, tld = get_clean_url(url)
        if tld:
            if clean_url.startswith("https"):
                clean_url = clean_url[8:]
                self.http_or_s["https"] += add_benign_or_malicious
            elif clean_url.startswith("http"):
                clean_url = clean_url[7:]
                self.http_or_s["http"] += add_benign_or_malicious
            if clean_url.startswith("www."):
                clean_url = clean_url[4:]
            path_to_file = url.replace(clean_url + ".{}".format(tld), "")
            self.tld_counter[tld] += add_benign_or_malicious

            ngrams_from_url = ngrams_string(3,self.len_ngrams,clean_url)

            for key in ngrams_from_url:
                self.ngrams_dict_domain[key] += add_benign_or_malicious
            if path_to_file:
                ngrams_path = ngrams_string(3,self.len_ngrams,path_to_file)
                for key in ngrams_path:
                    self.ngrams_dict_path[key] += add_benign_or_malicious


    def reduce(self):

        minimum_significance = self.data_len // self.threshold_for_removal
        self.ngrams_dict_domain = self.threshold_numpy_counter(self.ngrams_dict_domain, minimum_significance,True)
        self.ngrams_dict_path = self.threshold_numpy_counter(self.ngrams_dict_path, minimum_significance,True)
        chosen_ngrams = (self.ngrams_dict_path, self.ngrams_dict_domain,self.tld_counter, self.http_or_s, self.data_len,self.total_benign_malicious_vec)
        with open(self.pkl_path, "wb") as f_p:
            pickle.dump(chosen_ngrams,f_p)
        print("Saved {} as reduced pickle".format(self.pkl_path))

    def calc_chi_square(self):

        chi_square_results_benign = dict()
        chi_square_results_malicious = dict()
        self.all_data = {"tld":self.tld_counter,
                    "domain_ngrams": self.ngrams_dict_domain,
                    "pathes": self.ngrams_dict_path,
                         "http_or_s": self.http_or_s}
        for data_name, datatype in self.all_data.items():
            print("working on {}".format(data_name))
            for ngram, counts in tqdm.tqdm(datatype.items(), total=len(datatype)):
                chi_square_benign_and_malicious =self.calc_chi_square_specific_sample(counts)
                chi_square_results_benign[(data_name, ngram)] = chi_square_benign_and_malicious[0]
                chi_square_results_malicious[(data_name, ngram)] = chi_square_benign_and_malicious[1]

        #delattr(self, 'tld_counter')
        #delattr(self, 'ngrams_dict_domain')
        #delattr(self, 'ngrams_dict_path')
        self.chi_square_results_benign = chi_square_results_benign
        self.chi_square_results_malicious = chi_square_results_malicious
        self.select_according_to_chi_square()

    def calc_chi_square_specific_sample(self,counts):
        total_yes_feature = np.sum(counts)
        total_no_feature = self.data_len - total_yes_feature
        no_feature_counts = self.total_benign_malicious_vec - counts
        observed = np.concatenate((counts.reshape((1, 2)), no_feature_counts), axis=0)
        total_feature_vec = np.array((total_yes_feature, total_no_feature), dtype=np.uint32).reshape((2, 1))
        expected = np.matmul(total_feature_vec, self.total_benign_malicious_vec).astype(np.float32) / self.data_len
        return np.sum((observed - expected) ** 2 / expected, axis=0)
    def get_largest_chi_square(self,chi_square_results_benign):
        dataname_and_ngram, max_chi_square = max(chi_square_results_benign.items(), key=lambda kv: (kv[1], kv[0]))
        del chi_square_results_benign[dataname_and_ngram]
        data_name, ngram_added = dataname_and_ngram
        return data_name,ngram_added
    def select_according_to_chi_square(self):
        chosen_ngrams_malicious = defaultdict(dict)
        chosen_ngrams_benign = defaultdict(dict)
        for k in sorted(self.take_k_largest_chi_square):
            while(len(chosen_ngrams_benign)<k):
                data_name_benign,ngram_added_benign = self.get_largest_chi_square(self.chi_square_results_benign)
                chosen_ngrams_benign[data_name_benign][ngram_added_benign] = int(np.sum(self.all_data[data_name_benign][ngram_added_benign]))
                sub_ngrams = [key for key in self.all_data[data_name_benign] if key in ngram_added_benign and key is not ngram_added_benign]
                if len(sub_ngrams)>0:
                    self.all_data[data_name_benign][n_gram_] -= self.all_data[data_name_benign][
                        data_name_benign]

                    chi_square_benign_and_malicious = self.calc_chi_square_specific_sample(
                        self.all_data[data_name_benign][n_gram_])
                    self.chi_square_results_benign[(data_name_benign, n_gram_)] = chi_square_benign_and_malicious[0]
                    self.chi_square_results_malicious[(data_name_benign, n_gram_)] = chi_square_benign_and_malicious[
                        1]


                data_name_malicious,ngram_added_malicious = self.get_largest_chi_square(self.chi_square_results_malicious)
                chosen_ngrams_malicious[data_name_malicious][ngram_added_malicious] = int(np.sum(self.all_data[data_name_malicious][ngram_added_malicious]))
                sub_ngrams = [key for key in self.all_data[data_name_malicious] if key in ngram_added_malicious and key is not ngram_added_malicious]
                if len(sub_ngrams)>0:
                    for n_gram_ in sub_ngrams:
                        self.all_data[data_name_malicious][n_gram_] -=self.all_data[data_name_malicious][ngram_added_malicious]

                        chi_square_benign_and_malicious = self.calc_chi_square_specific_sample(self.all_data[data_name_malicious][n_gram_] )
                        self.chi_square_results_benign[(data_name_malicious, n_gram_)] = chi_square_benign_and_malicious[0]
                        self.chi_square_results_malicious[(data_name_malicious, n_gram_)] = chi_square_benign_and_malicious[1]

        chosen_ngrams = {key:{**chosen_ngrams_malicious[key],**chosen_ngrams_benign[key]} for key in set(list(chosen_ngrams_malicious)).union(set(list(chosen_ngrams_benign)))}
        with open(os.path.join(path_data, "important_ngrams_split_{}_size={}.json".format(self.len_ngrams,k)), "w") as f_p:
            json.dump(chosen_ngrams, f_p)

    def add_ngrams(self,chi_square_results_benign,benign_chi_square_chosen,chosen_ngrams):
        for chi_square_score in benign_chi_square_chosen:
            data_names, ngrams_s = zip(*chi_square_results_benign[chi_square_score])
            if len(set(ngrams_s)) < len(ngrams_s):
                print("handle this case")
            for data_name, ngram_added in chi_square_results_benign[chi_square_score]:
                chosen_ngrams[data_name][ngram_added] = int(np.sum(self.all_data[data_name][ngram_added]))
                sub_ngrams = [key for key in self.all_data[data_name] if key in ngram_added and key is not ngram_added]
                if len(sub_ngrams)>0:
                    print(sub_ngrams)
        return chosen_ngrams
if __name__ == "__main__":

    weights_corr = 1e-5
    debug = True

    if debug:
        path_data = "/mnt/nimble_storage/ilan/data/Balanced_debug_02_10_2022_08_19_56/validation_data"
        meaningul_ngrams = Meaninigfull_Ngrams(path_data, 5)
    else:
        path_data = "/mnt/nimble_storage/ilan/data/Balanced_02_10_2022_08_19_56/train_data"
        meaningul_ngrams = Meaninigfull_Ngrams(path_data, 7)

    if meaningul_ngrams.load_pickle(path_data):
        pass
    else:
        meaningul_ngrams.run_all()
        meaningul_ngrams.reduce()

    meaningul_ngrams.calc_chi_square()



    # all_set = {"."+tld__ for tld__ in tld_list}
    with open(os.path.join(path_data, "important_ngrams_split_{}.json".format(len_ngrams)), "w") as f_p:
        json.dump(all_data2, f_p)
    print("number of n-grams in domain: {}".format(len(ngrams_list_domain)))
    print("number of n-grams in path: {}".format(len(ngrams_list_path)))
    print("number of tlds: {}".format(len(tld_list)))

    print("Chi-square ngrams")

    all_set = all_set.union(set(ngrams_list_domain)).union(set(ngrams_list_domain))

    with open("ngrams_train.json", "w") as f_p:
        json.dump(list(all_set), f_p)
    print("json was written")
    tld_dict = {key: k for k, key in enumerate(tld_list)}
    tld_list.append("")

    num_of_samples = 200
    counter_ = 0

    for dataset_name in all_dss:
        ds = {"url": [], "label": []}

        ds_domain = []
        ds_tld = []
        ds_path = []
        for url, label in tqdm.tqdm(all_dss[dataset_name][["redirected", "label"]].itertuples(index=False),
                                    total=len(all_dss[use_for_training])):
            url = urllib.parse.unquote(url)
            tld_res = np.zeros(len(tld_list), bool)
            url_res = np.zeros(len(ngrams_list_domain), dtype=bool)
            path_res = np.zeros(len(ngrams_list_path), dtype=bool)

            try:
                clean_url, tld = get_clean_url(url)
                if tld:
                    tld_ind = tld_dict[tld]
                    tld_res[tld_ind] = True

                    path_to_file = url.replace(clean_url + ".{}".format(tld), "")
                    ngrams_from_url = {"".join(n_gram) for i in range(2, len_ngrams) for n_gram in
                                       ngrams(clean_url, i + 1)}
                    indices_from_url = np.array(
                        [ngrams_dict_domain[key] for key in ngrams_from_url if key in ngrams_dict_domain], dtype=int)
                    url_res[indices_from_url] = True

                else:
                    tld_res[-1] = True
                    path_to_file = url.replace(clean_url, "")

                if path_to_file:
                    ngrams_path = {"".join(n_gram) for i in range(2, len_ngrams) for n_gram in
                                   ngrams(path_to_file, i + 1)}
                    indices_from_url = np.array(
                        [ngrams_dict_path[key] for key in ngrams_dict_path if key in ngrams_dict_path], dtype=int)
                    path_res[indices_from_url] = True

            except:
                continue
            ds_tld.append(tld_res)
            ds_domain.append(url_res)
            ds_path.append(path_res)
            ds["url"].append(url)
            ds["label"].append(label)
            if (len(ds_path) % num_of_samples) == 0:
                print("creating ds")
                df = pd.DataFrame(ds)
                df["label"] = df["label"].astype(bool)
                print("creating domain")
                ds_domain = pd.DataFrame(np.array(ds_domain, dtype=bool),
                                         columns=["domain:{}".format(key) for key in ngrams_list_domain])
                print("creating tlds")
                ds_tld = pd.DataFrame(np.array(ds_tld, dtype=bool), columns=["tld:{}".format(key) for key in tld_list])
                print("creating path")
                ds_path = pd.DataFrame(np.array(ds_path, dtype=bool),
                                       columns=["path:{}".format(key) for key in ngrams_list_path])
                print("joining")
                all_ds = pd.concat([df, ds_domain, ds_tld, ds_path], axis=1)
                print("to parquet {}".format(counter_))

                all_ds.to_csv("{}_{}.csv".format(dataset_name, counter_), compression=None)
                ds_tld = []
                ds_domain = []
                ds_path = []
                ds = {"url": [], "label": []}
                counter_ += 1
