from geo_llama.main import GeoLlama
from geo_llama.model import TopoModel, RAGModel
from argparse import ArgumentParser

"""Runs the model on the same peice of text N times, allowing users to 
view the degree of uncertainty in the model predictions. 
"""
def load_geo_llama():
    # load the geollama model
    topo_model = TopoModel(model_name='JoeShingleton/GeoLlama-3.1-8b-toponym', 
                    prompt_path='data/prompt_templates/prompt_template.txt',
                    instruct_path='data/prompt_templates/topo_instruction.txt',
                    input_path=None,
                    config_path='data/config_files/model_config.json')

    rag_model = RAGModel(model_name='JoeShingleton/GeoLlama-3.1-8b-RAG', 
                        prompt_path='data/prompt_templates/prompt_template.txt',
                        instruct_path='data/prompt_templates/rag_instruction.txt',
                        input_path='data/prompt_templates/rag_input.txt',
                        config_path='data/config_files/model_config.json')

    return GeoLlama(topo_model, rag_model)
    
def repeated_prediction(text, count=100):
    
    toponym_predictions = []
    for i in range(100):
        toponym_predictions.append(geo_llama.geoparse(text))
    
    return toponym_predictions
 
    

if __name__ == '__main__':
    
    # get the filename from parse    
    parser = ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, dest='filename')
    parser.add_argument('-c', '--count', dest='count')
    args = parser.parse_args()
    
    # load the geollama model
    geo_llama = load_geo_llama()
    
    # read the article to be parsed    
    with open(args.filename, 'r') as f:
        text = f.read()
        
    # parse 100 times
    pred_articles = repeated_prediction(text, count=args.count)
    
    article_name = args.filename.split('/')[-1].split('.')[0]
    with open(f'geo_llama_{article_name}_n{args.count}.json', 'w') as f:
        f.dump(pred_articles)
        