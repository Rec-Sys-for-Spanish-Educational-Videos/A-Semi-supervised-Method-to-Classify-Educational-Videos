import requests
import wikipedia
import urllib

wikipedia.set_lang("es")

S = requests.Session()
URL = "https://es.wikipedia.org/w/api.php"
maxDepth = 1
visited = {}

labelledData=[]


def findPagesinSubcategories(pageName, depth,label):

    if(depth < maxDepth):
        visited[pageName] = 1

        url = urllib.parse.unquote(wikipedia.page(pageName).url)
        
      

        categoryName = url.split('/')[-1]
        print(categoryName)
        PARAMS = {
            'action': "query",
            'list': "categorymembers",
            'cmtitle': categoryName,
            'cmlimit': 500,
            'format': "json",
            'cmtype': "page"
        }

        R = S.get(url=URL, params=PARAMS)
        data= R.json()
        query = data['query']
        category = query['categorymembers']

        for x in category:
               try:
                   labelledData.append([x['title'], wikipedia.page(x['title']).content, label])
                   print("Extracted page " + x['title'] )
               except:
                   print(x['title'] + " page doesn't exist!")


        PARAMS = {
            'action': "query",
            'list': "categorymembers",
            'cmtitle': categoryName,
            'cmlimit': 500,
            'format': "json",
            'cmtype': "subcat"
        }

        R = S.get(url=URL, params=PARAMS)
        data= R.json()
        
        query = data['query']
        category = query['categorymembers']
        for x in category:
            if(x['title'] not in visited and x['title'].count(':') < 2):
                 findPagesinSubcategories(x['title'], depth + 1, 0)
        
        

findPagesinSubcategories('Categoría:Biología',0,0)
findPagesinSubcategories('Categoría:Anatomía',0,0)
findPagesinSubcategories('Categoría:Bioinformática‎',0,0)
findPagesinSubcategories('Categoría:Biología celular‎',0,0)
findPagesinSubcategories('Categoría:Bioquímica‎',0,0)
findPagesinSubcategories('Categoría:Biotecnología‎',0,0)
findPagesinSubcategories('Categoría:Botánica‎',0,0)
findPagesinSubcategories('Categoría:Microbiología‎',0,0)
findPagesinSubcategories('Categoría:Genética‎',0,0)

findPagesinSubcategories('Categoría:Ingeniería',0,1)
findPagesinSubcategories('Categoría:Materiales en ingeniería',0,1)
findPagesinSubcategories('Categoría:Bases de datos‎',0,1)
findPagesinSubcategories('Categoría:Computación distribuida‎',0,1)
findPagesinSubcategories('Categoría:Computación gráfica‎',0,1)
findPagesinSubcategories('Categoría:Geomática‎',0,1)
findPagesinSubcategories('Categoría:Ingeniería de software‎',0,1)
findPagesinSubcategories('Categoría:Seguridad informática‎',0,1)


findPagesinSubcategories('Categoría:Arte',0,2)
findPagesinSubcategories('Categoría:Técnicas_de_arte',0,2)
findPagesinSubcategories('Categoría:Antropología',0,2)
findPagesinSubcategories('Categoría:Símbolos',0,2)
findPagesinSubcategories('Categoría:Ciencias_Históricas',0,2)

findPagesinSubcategories('Categoría:Ciencias_sociales',0,3)
findPagesinSubcategories('Categoría:Economía',0,3)
findPagesinSubcategories('Categoría:Sociología',0,3)
findPagesinSubcategories('Categoría:Comunicación',0,3)
findPagesinSubcategories('Categoría:Términos_jurídicos',0,3)
findPagesinSubcategories('Categoría:Justicia',0,3)
findPagesinSubcategories('Categoría:Derecho',0,3)
findPagesinSubcategories('Categoría:Principios_del_derecho',0,3)






   
