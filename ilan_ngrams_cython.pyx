
cdef ngrams_ilan(set(cdef str sequence,cdef int start,cdef int end)):
    end = min(end,len(sequence))
    output_set = {sequence[ind_start:ind_start+len_ngram] for len_ngram in range(start,end) for ind_start in range(len(sequence)-len_ngram+1)}
    return output_set