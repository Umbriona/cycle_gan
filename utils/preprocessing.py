import numpy as np
from Bio import SeqIO
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf

def to_binary(seq, max_length, start_stop = False):
    # eoncode non-standard amino acids like X as all zeros
    # output a array with size of L*20
    seq = seq.upper()
    if not start_stop:
        aas = 'ACDEFGHIKLMNPQRSTVWYX'
        vocab=21
    else:
        aas = 'ACDEFGHIKLMNPQRSTVWYX<>'
        vocab=23
    pos = dict()
    for i in range(len(aas)): pos[aas[i]] = i
    
    binary_code = dict()
    for aa in aas: 
        code = np.zeros(vocab, dtype = np.float32)
        code[pos[aa]] = 1
        binary_code[aa] = code
    
    seq_coding = np.zeros((max_length,vocab), dtype = np.float32)
    for i,aa in enumerate(seq): 
        code = binary_code.get(aa,np.zeros(vocab, dtype = np.float32))
        seq_coding[i,:] = code
    return seq_coding

def to_int(seq, max_length, start_stop = False):
    seq.upper()
    if not start_stop:
        aas = 'ACDEFGHIKLMNPQRSTVWYX'
    else:
        aas = 'ACDEFGHIKLMNPQRSTVWYX<>'
  
    d = dict()
    for i in range(len(aas)): d[aas[i]] = i

    tmp =np.array([d[i] if i in aas else 20 for i in seq])
    out = np.ones((max_length,))*20
    index = tmp.size if tmp.size<max_length else max_length
    out[:index] = tmp[:index]
    return out

def loss_weight(mask, max_length):
    len_seq = len(mask)
    seq_w = [1 for i in mask] 
    tmp = np.ones((max_length,))
    tmp[:len_seq] = seq_w
    tmp[len_seq:] = 0.0
    return tmp




    
def zero_padding(inp,length=500,start=False):
    # zero pad input one hot matrix to desired length
    # start .. boolean if pad start of sequence (True) or end (False)
    #assert len(inp) <= length
    out = np.zeros((length,inp.shape[1]))
    if start:
        out[-inp.shape[0]:] = inp[:length,:]
    else:
        out[0:inp.shape[0]] = inp[:length,:]
    return out

def prepare_dataset(file_path, file_name, file_format = 'fasta', seq_length = 1024, t_v_split = 0.1,start_stop = False, max_samples = 5000):
    
    if start_stop:
        seq_length -= 2

    count=0
    dict_ = {'id':[] ,'mask':[],'seq':[], 'mask_bin':[], 'seq_bin':[], 'loss_weight':[], 'seq_int':[]}
    # loading data to dict
    for i, rec in enumerate(SeqIO.parse(os.path.join(file_path,file_name),'fasta')):
        count +=1
        if count >max_samples:
            break
        if len(rec.seq)>seq_length:
            continue
        dict_['id'].append(rec.id)
        if not start_stop:
            dict_['seq'].append(str(rec.seq))
            dict_['loss_weight'].append(loss_weight(rec.seq ,seq_length))
            dict_['seq_int'].append(to_int(rec.seq, max_length=seq_length))
            dict_['seq_bin'].append(to_binary(rec.seq, max_length=seq_length))
        else:
            str_seq = '<'+str(rec.seq)+'>'
            dict_['seq'].append(str(rec.seq))
            dict_['loss_weight'].append(loss_weight(str_seq ,seq_length+2))
            dict_['seq_int'].append(to_int(str_seq, max_length=seq_length+2, start_stop=start_stop))
            dict_['seq_bin'].append(to_binary(str_seq, max_length=seq_length+2, start_stop=start_stop))
   # Splitting data to training and validation sets

    int_train, int_test, W_train, W_test, bin_train, bin_test, id_train, id_test = train_test_split(np.array(dict_['seq_int'],dtype=np.int8),
                                                    np.array(dict_['loss_weight'],dtype=np.float32),
                                                    np.array(dict_['seq_bin'], dtype = np.float32),
                                                    dict_['id'],
                                                    test_size=t_v_split, random_state=42)
   
    dataset_train = tf.data.Dataset.from_tensor_slices((int_train,bin_train,W_train))
    dataset_validate = tf.data.Dataset.from_tensor_slices((int_test,bin_test,W_test))
    return dataset_train, dataset_validate
    
def prepare_dataset_U_N(file_path, file_name, file_format = 'fasta', seq_length = 1024, t_v_split = 0.1):
    

    count=0
    dict_ = {'id':[] ,'mask':[],'seq':[], 'mask_bin':[], 'seq_bin':[], 'loss_weight':[], 'seq_int':[]}
    # loading data to dict
    for i, rec in enumerate(SeqIO.parse(os.path.join(file_path,file_name),'fasta')):
        count +=1
        if count >10000:
            break
        if len(rec.seq)>seq_length:
            continue
        dict_['id'].append(rec.id)
        dict_['seq'].append(rec.seq)
        dict_['loss_weight'].append(loss_weight(rec.seq ,seq_length))
        dict_['seq_int'].append(to_int(rec.seq, max_length=seq_length))
        dict_['seq_bin'].append(to_binary(rec.seq, max_length=seq_length))
   # Splitting data to training and validation sets

    int_train, int_test, W_train, W_test, bin_train, bin_test, id_train, id_test = train_test_split(np.array(dict_['seq_int'],dtype=np.int8),
                                                    np.array(dict_['loss_weight'],dtype=np.float32),
                                                    np.array(dict_['seq_bin'], dtype = np.float32),
                                                    dict_['id'],
                                                    test_size=t_v_split, random_state=42)
   
    dataset_train = tf.data.Dataset.from_tensor_slices((int_train,bin_train,W_train))
    dataset_validate = tf.data.Dataset.from_tensor_slices((int_test,bin_test,W_test))
    return dataset_train, dataset_validate