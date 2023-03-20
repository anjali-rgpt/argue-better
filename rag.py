import os
from tqdm import tqdm
import argparse
import pandas as pd
import json

from fastrag.stores import PLAIDDocumentStore
from fastrag.retrievers.colbert import ColBERTRetriever
from haystack.nodes import SentenceTransformersRanker
from fastrag.readers import T5Reader
from fastrag.readers.FiD import FiDReader
from fastrag.utils import get_timing_from_pipeline
from haystack import Pipeline

def jsonify(res, reader=1, k=1, j=3):

    output = {
        'query': res['query'],
        'ans': dict(),
        'docs': dict()
    }
        
    if reader:
        ans = res['answers'][:k]
        for m, a in enumerate(ans):
            output['ans'][m] = {
                'answer': a.answer,
                'score': a.score,
                'context': a.context
            }

    docs = res['documents'][:j]
    for n, d in enumerate(docs):
        output['docs'][n] = {
            'content': d.content,
            'id': d.id,
            'score': d.score,
            'meta': d.meta 
        }

    return output


def main():
    parser = argparse.ArgumentParser("Create an index using PLAID engine as a backend")
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--generative", type=int, default=1)
    args = parser.parse_args()
    
    dataroot = 'data'
    dataset = 'effective'
    datasplit = 'train'

    queries_path = os.path.join(dataroot, dataset, datasplit, 'questions.search.tsv')
    collection_path = os.path.join(dataroot, dataset, datasplit, 'collection.tsv')

    queries = pd.read_csv(queries_path, sep='\t', header=None)

    nbits = 2
    create = True if args.index else False
    index_name = f'{dataset}.{datasplit}.{nbits}bits'
    
    #---------------------#
    #  create components  #
    #---------------------#
    store = PLAIDDocumentStore(
        index_path=index_name,
        checkpoint_path="Intel/ColBERT-NQ",
        collection_path=collection_path,
        create=create,
        nbits=nbits,
        gpus=args.gpus,
        ranks=args.ranks,
        doc_maxlen=120,
        query_maxlen=60,
        kmeans_niters=4,
    )
    
    retriever = ColBERTRetriever(
        store, 
        top_k=100 if args.generative else 10
    )
    reranker = SentenceTransformersRanker(
        model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2", 
        top_k=10
    )

    # reader = T5Reader(
    #     model_name_or_path="google/flan-t5-base", 
    #     input_converter_tokenizer_max_len=16300,  
    #     min_length=50, 
    #     max_length=500, 
    #     num_beams=4, 
    #     top_k=1, 
    #     use_gpu=True
    # )

    reader = FiDReader(
        input_converter_tokenizer_max_len=250,
        max_length=20,
        model_name_or_path="path/to/fid",
        use_gpu=True
    )

    #--------------------#
    #   build pipeline   #
    #--------------------#

    p = Pipeline()

    p.add_node(component=retriever, name="Retriever", inputs=["Query"])
    if args.generative:
        p.add_node(component=reranker, name="Reranker", inputs=["Retriever"])
        p.add_node(component=reader, name="Reader", inputs=["Reranker"])

    #------------------#
    #   run pipeline   #
    #------------------#

    # store top 3 retrieved docs
    results = dict()
    for i in tqdm(range(len(queries))):
        res = p.run(query=queries[1][i])
        results[i] = jsonify(res, reader=args.generative)
    
    # output results into json file
    json_output = json.dumps(results, indent=4) 
    with open("data/effective/res_t5.json", "w") as outfile:
        outfile.write(json_output)
    

if __name__ == "__main__":
    main()
