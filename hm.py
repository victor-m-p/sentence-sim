# sentenceBERT
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from datetime import datetime 

####### actual thing now #######
## do 10x10 and then a heatmap ## 
def read_sentences(infile):
    with open(f"{infile}", "r") as f: 
        sentences = f.read().splitlines() 
    return sentences 

# cosine similarity (check native approaches to BERT?)
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

## run all pair-wise similarities
### really should only do this for one way (i.e. we do double work here)
def compute_cosine_sim(sentences, cosine, model): 
    dct = {}
    for sentence in sentences: 
        print(f"sentence: {sentence}")
        for query in sentences: 
            print(f"query: {query}")
            sim = cosine(model.encode([sentence])[0], model.encode([query])[0])
            dct[(sentence, query)] = sim
    return dct

## dictionary to pandas
def reshape_dct(dct, decimals = 2):

    # to pandas 
    df = pd.Series(dct).reset_index()   
    df.columns = ['w1', 'w2', 'sim']

    ## reshape pandas and prepare heatmap
    df_unstacked = df.set_index(['w1', 'w2']).unstack()
    df_unstacked.columns = df_unstacked.columns.droplevel()
    x = list(df_unstacked.columns)
    y = list(df_unstacked.index)
    vals = df_unstacked.to_numpy()
    vals = vals.round(decimals = decimals) # round decimals

    return x, y, vals

def plot_hm(x, y, vals, outpath, figsize = (10, 10))

    ## plot 
    fig, ax = plt.subplots(figsize = figsize)
    im = ax.imshow(vals)

    # show ticks and labels 
    ax.set_xticks(np.arange(len(x)), labels = x)
    ax.set_yticks(np.arange(len(y)), labels = y)

    # rotate ticks and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode = "anchor")

    # loop over elements and create annotations
    for i in range(len(y)):
        for j in range(len(x)):
            text = ax.text(j, i, vals[i, j],
                        ha="center", va="center", color="w")

    # save 
    timestamp = datetime.now().strftime("%H-%M-%S")
    plt.savefig(f"{outpath}/hm_{timestamp}.png") # should be pdf

# main
def main(infile, outpath): 

    ## import sentenceBERT model & instantiate
    print("instantiating model")
    model = SentenceTransformer('bert-base-nli-mean-tokens')

    ## read sentences 
    print("reading sentences")
    sentences = read_sentences(infile)

    ## compute distance 
    print("computing cosine distance with sentenceBERT")
    dct = compute_cosine_sim(sentences, cosine, model)

    ## reshape
    print(f"reshape results")
    x, y, vals = reshape_dct(dct) # decimals = 2 baseline

    ## plot 
    plot_hm(x, y, vals, outpath) # figsize = (10, 10) baseline

# setup 
