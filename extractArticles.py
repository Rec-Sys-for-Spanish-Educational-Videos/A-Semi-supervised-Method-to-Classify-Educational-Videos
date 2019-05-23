import requests
import wikipedia
import urllib

class WikipediaExtractor:
    
    def __init__(self, maxDepth, language):
        wikipedia.set_lang(language)
        
        self.S = requests.Session()
        self.URL = "https://" + language +".wikipedia.org/w/api.php"
        self.maxDepth = maxDepth
        self.visited = {}
        
        self.callback = None
        
        self.labelledData=[]
        
    def setCallbackFunction(self, callback):
        self.callback = callback
        
    def findNrPagesinSubcategories(self, pageName,label):
        self.visisted = {}
        
        return self.findNrPages(pageName, 0,label)

    def findNrPages(self, pageName, depth,label): 
        nr = 0
        if(depth < self.maxDepth):
            self.visited[pageName] = 1
    
            url = urllib.parse.unquote(wikipedia.page(pageName).url)
            
          
    
            categoryName = url.split('/')[-1]
         
            PARAMS = {
                'action': "query",
                'list': "categorymembers",
                'cmtitle': categoryName,
                'cmlimit': 500,
                'format': "json",
                'cmtype': "page"
            }
    
            R = self.S.get(url=self.URL, params=PARAMS)
            data= R.json()
            query = data['query']
            category = query['categorymembers']
    
            nr+= len(category)
    
    
            PARAMS = {
                'action': "query",
                'list': "categorymembers",
                'cmtitle': categoryName,
                'cmlimit': 500,
                'format': "json",
                'cmtype': "subcat"
            }
    
            R = self.S.get(url=self.URL, params=PARAMS)
            data= R.json()
            
            query = data['query']
            category = query['categorymembers']
            for x in category:
                if(x['title'] not in self.visited and x['title'].count(':') < 2):
                    nr+=self.findNrPages(x['title'], depth + 1, 0)
            
        return nr
        
      
    def findPagesinSubcategories(self,pageName,label):
        self.visisted = {}
        
        self.findPages(pageName, 0,label)

    def findPages(self, pageName, depth,label):
    
        if(depth < self.maxDepth):
            self.visited[pageName] = 1
    
            url = urllib.parse.unquote(wikipedia.page(pageName).url)
            
          
    
            categoryName = url.split('/')[-1]
          
            PARAMS = {
                'action': "query",
                'list': "categorymembers",
                'cmtitle': categoryName,
                'cmlimit': 500,
                'format': "json",
                'cmtype': "page"
            }
    
            R = self.S.get(url=self.URL, params=PARAMS)
            data= R.json()
            query = data['query']
            category = query['categorymembers']
    
            for x in category:
                   try:
                       self.labelledData.append([x['title'], wikipedia.page(x['title']).content, label])
                       
                       if self.callback is not None:
                           self.callback(True, x['title'])
                   except:
                       if self.callback is not None:
                           self.callback(False, x['title'])
                     
    
    
            PARAMS = {
                'action': "query",
                'list': "categorymembers",
                'cmtitle': categoryName,
                'cmlimit': 500,
                'format': "json",
                'cmtype': "subcat"
            }
    
            R = self.S.get(url=self.URL, params=PARAMS)
            data= R.json()
            
            query = data['query']
            category = query['categorymembers']
            for x in category:
                if(x['title'] not in self.visited and x['title'].count(':') < 2):
                     self.findPages(x['title'], depth + 1, 0)
        
        







   
