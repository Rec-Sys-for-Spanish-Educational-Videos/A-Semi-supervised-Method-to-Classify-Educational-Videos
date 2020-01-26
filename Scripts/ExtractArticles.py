import requests
import wikipedia
import urllib
import pandas
import numpy as np
from collections import deque

SESSION = requests.Session()
URL = "https://es.wikipedia.org/w/api.php"

wikipedia.set_lang('es')

def get_pages_from_category(page_id):
    
    PARAMS = {
        'action': "query",
        'list': "categorymembers",
        "cmpageid":int(page_id),
        'cmlimit': 500,
        'format': "json",
        'cmtype': "page"
    }

    result = SESSION.get(url = URL, params = PARAMS).json();

    return  result['query']['categorymembers']

def get_subcategories_from_category(page_id):

    PARAMS = {
        'action': "query",
        'list': "categorymembers",
        'cmpageid': int(page_id),
        'cmlimit': 500,
        'format': "json",
        'cmtype': "subcat"
    }

    result = SESSION.get(url = URL, params = PARAMS).json();

    return  result['query']['categorymembers']


maxDepth = 1
visited = {}

labelledData=[]

def findPagesinSubcategories(page_id, depth,label):
    if(depth < maxDepth):
        visited[page_id] = 1
    
        pages = get_pages_from_category(page_id)

        for page in pages:
               try:
                   page_info = wikipedia.page(pageid = page['pageid'])
                   links =  page_info.links
                   labelledData.append([page['pageid'], page['title'], page_info.content,str(links), label])
                   print("Extracted page " + page['title'] )
               except:
                   print(page['title'] + " page doesn't exist!")


        subcategories = get_subcategories_from_category(page_id)

        for subcategory in subcategories:
            if(subcategory['pageid'] not in visited and subcategory['title'].count(':') < 2):
                 findPagesinSubcategories(subcategory['pageid'], depth + 1, 0)
        

findPagesinSubcategories(wikipedia.page('Categoría:Biología').pageid,0,0)
findPagesinSubcategories(wikipedia.page('Categoría:Anatomía').pageid,0,0)
findPagesinSubcategories(wikipedia.page('Categoría:Bioinformática‎').pageid,0,0)
findPagesinSubcategories(wikipedia.page('Categoría:Biología celular‎').pageid,0,0)
findPagesinSubcategories(wikipedia.page('Categoría:Bioquímica‎').pageid,0,0)
findPagesinSubcategories(wikipedia.page('Categoría:Biotecnología‎').pageid,0,0)
findPagesinSubcategories(wikipedia.page('Categoría:Botánica‎').pageid,0,0)
findPagesinSubcategories(wikipedia.page('Categoría:Microbiología‎').pageid,0,0)
findPagesinSubcategories(wikipedia.page('Categoría:Genética‎').pageid,0,0)

findPagesinSubcategories(wikipedia.page('Categoría:Ingeniería').pageid,0,1)
findPagesinSubcategories(wikipedia.page('Categoría:Materiales en ingeniería').pageid,0,1)
findPagesinSubcategories(wikipedia.page('Categoría:Bases de datos‎').pageid,0,1)
findPagesinSubcategories(wikipedia.page('Categoría:Computación distribuida‎').pageid,0,1)
findPagesinSubcategories(wikipedia.page('Categoría:Computación gráfica‎').pageid,0,1)
findPagesinSubcategories(wikipedia.page('Categoría:Geomática‎').pageid,0,1)
findPagesinSubcategories(wikipedia.page('Categoría:Ingeniería de software‎').pageid,0,1)
findPagesinSubcategories(wikipedia.page('Categoría:Seguridad informática‎').pageid,0,1)


findPagesinSubcategories(wikipedia.page('Categoría:Arte').pageid,0,2)
findPagesinSubcategories(wikipedia.page('Categoría:Técnicas_de_arte').pageid,0,2)
findPagesinSubcategories(wikipedia.page('Categoría:Antropología').pageid,0,2)
findPagesinSubcategories(wikipedia.page('Categoría:Símbolos').pageid,0,2)
findPagesinSubcategories(wikipedia.page('Categoría:Ciencias_Históricas').pageid,0,2)
findPagesinSubcategories(wikipedia.page('Categoría:Ciencias_sociales').pageid,0,2)
findPagesinSubcategories(wikipedia.page('Categoría:Economía').pageid,0,2)
findPagesinSubcategories(wikipedia.page('Categoría:Sociología').pageid,0,2)
findPagesinSubcategories(wikipedia.page('Categoría:Comunicación').pageid,0,2)
findPagesinSubcategories(wikipedia.page('Categoría:Términos_jurídicos').pageid,0,2)
findPagesinSubcategories(wikipedia.page('Categoría:Justicia').pageid,0,2)
findPagesinSubcategories(wikipedia.page('Categoría:Derecho').pageid,0,2)
findPagesinSubcategories(wikipedia.page('Categoría:Principios_del_derecho').pageid,0,2)

labelledData = np.array(labelledData)

df = pandas.DataFrame.from_dict({'Id': labelledData[:,:1].tolist(),'Title': labelledData[:,1:2].tolist(),
                            'Content':labelledData[:,2:3].tolist(), 'Keywords':labelledData[:,3:4].tolist(), 
                            'Label':labelledData[:,4:5].tolist()})

df.to_excel('WikipediaTruthDataset.xlsx', header=True, index=False)


   
