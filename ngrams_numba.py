import numpy as np
import numba as nb
from numba import njit,prange,jit
from numba.typed import Dict
from numba.types import bool_
#value_type = nb.types.Array(nb.types.int8,100,"A")
# @njit
# def ngrams_list(ngrams_min=3, ngrams_max=7,txt_of_texts=["",""]):
#      res = Dict.empty(key_type=nb.types.unicode_type,value_type=value_type)
#
#      for i in range(len(txt_of_texts)):
#          txt = txt_of_texts[i]
#          ngrams_this = max(ngrams_max, len(txt))
#          for len_of_str in range(ngrams_min,ngrams_this):
#              for string_start in range(0,len(txt)-len_of_str):
#                  str1 = txt[string_start:string_start+len_of_str]
#                  res[str1][i] = 1
#
#      return res

@njit
def ngrams_string(ngrams_min=3, ngrams_max=7,txt=""):
     res = Dict.empty(key_type=nb.types.unicode_type,value_type=nb.types.int64)
     ngrams_this = max(ngrams_max, len(txt))
     for len_of_str in range(ngrams_min,ngrams_this):
         for string_start in range(0,len(txt)-len_of_str):
             str1 = txt[string_start:string_start+len_of_str]
             res[str1] = 1



     return res
#print(ngrams_list(3,7,["abcdefgh","abcababab"]))
print(ngrams_string(3,7,"abcababab"))