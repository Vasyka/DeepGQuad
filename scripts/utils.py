import torch
import numpy as np
import pandas as pd

# Center te sequences
def center_seq(max_seq_len = 500):

    gquad_places = pd.read_csv('G4_chip.bed', sep='\t', header=None, names = ['chr','start','end'])
    gquad_places['len'] = gquad_places.end - gquad_places.start
    print('Start shape:', gquad_places.shape)
    
    gquad_places['new_start'] = 0
    gquad_places['new_end'] = 0
    for i, row in gquad_places.iterrows():
        if row.len <= max_seq_len:
            shift_l = int((max_seq_len - row.len) / 2)
            shift_r = int((max_seq_len - row.len) / 2 + 0.5)
            gquad_places.loc[i,'new_start'] = row.start - shift_l
            gquad_places.loc[i,'new_end'] = row.end + shift_r
        else:
            gquad_places = gquad_places.drop(i, axis=0)
    print('Final shape:',gquad_places.shape)
    gquad_places[['chr','new_start','new_end']].to_csv('G4_chip_centered.bed', sep='\t',header=False, index = False)
    return gquad_places

# Load all data and merge chromosomes in 1 table
def merge_chrom_data(file_name_start, chromosomes, file_name_end = ''):
    columns = ['chr','start','end','seq']
    data = pd.DataFrame()
    for num in chromosomes:
        chrom = pd.read_csv(f'{file_name_start}{num}{file_name_end}.bed',sep='\t',header=None)
        data = pd.concat([data, chrom])
    data.columns=columns
    data = data.reset_index(drop=True)
    data['len'] = data.end - data.start

    print(data.chr.unique())
    return data

# One-hot encoding of sequence
def encode_seq(sequences):
    encoder = {'A':[0,0,0,1],'T':[0,0,1,0],'G':[0,1,0,0],'C':[1,0,0,0],'N':[0,0,0,0]}
    
    encoded_seqs = []
    for seq in sequences:
        encoded_seq = list(map(lambda nucl: encoder[nucl], seq.upper()))
        encoded_seqs.append(encoded_seq)
    return encoded_seqs

# Select what to use - cpu\cuda
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

