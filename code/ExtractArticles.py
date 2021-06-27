import requests
import wikipedia
import urllib

wikipedia.set_lang("en")

S = requests.Session()
URL = "https://en.wikipedia.org/w/api.php"
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
        isIn = "query" in data
        if isIn :
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
                     findPagesinSubcategories(x['title'], depth + 1, label)


#category 0 - biology
findPagesinSubcategories('Category:Natural_sciences‎',0,0)  
findPagesinSubcategories('Category:Biology',0,0)  
findPagesinSubcategories('Category:Botany‎',0,0)  
findPagesinSubcategories('Category:Ecology‎',0,0)
findPagesinSubcategories('Category:Medicine‎',0,0)  
findPagesinSubcategories('Category:Neuroscience‎',0,0)  
findPagesinSubcategories('Category:Chemistry‎',0,0)  
findPagesinSubcategories('Category:Plants‎',0,0)  
findPagesinSubcategories('Category:Pollution',0,0)  


 #category 1 - engineering
findPagesinSubcategories('Category:Formal_sciences‎',0,1)  
findPagesinSubcategories('Category:Mathematics‎',0,1)  
findPagesinSubcategories('Category:Mathematics_education‎',0,1)  
findPagesinSubcategories('Category:Equations',0,1)  
findPagesinSubcategories('Category:Heuristics‎',0,1)  
findPagesinSubcategories('Category:Measurement‎',0,1)
findPagesinSubcategories('Category:Numbers‎',0,1)  
findPagesinSubcategories('Category:Mathematical_proofs‎',0,1)  
findPagesinSubcategories('Category:Theorems‎',0,1)  
findPagesinSubcategories('Category:Fields_of_mathematics',0,1)  
findPagesinSubcategories('Category:Statistics‎',0,1)  
findPagesinSubcategories('Category:Analysis_of_variance',0,1)  
findPagesinSubcategories('Category:Categorical_data‎',0,1)  
findPagesinSubcategories('Category:Data_analysis‎',0,1)  
findPagesinSubcategories('Category:Decision_theory',0,1)  
findPagesinSubcategories('Category:Statistical_theory‎',0,1)  
findPagesinSubcategories('Category:Survival_analysis‎',0,1)

 #category 2 - humanities
findPagesinSubcategories('Category:Health_care_occupations‎',0,2)
findPagesinSubcategories('Category:History‎',0,2)  
findPagesinSubcategories('Category:Events‎',0,2)  
findPagesinSubcategories('Category:Activism‎',0,2)  
findPagesinSubcategories('Category:Agriculture‎',0,2)
findPagesinSubcategories('Category:The_arts‎',0,2)  
findPagesinSubcategories('Category:Aviation‎',0,2)  
findPagesinSubcategories('Category:Commemoration‎',0,2)  
findPagesinSubcategories('Category:Communication‎',0,2)
findPagesinSubcategories('Category:Crime‎',0,2)  
findPagesinSubcategories('Category:Design‎',0,2)  
findPagesinSubcategories('Category:Education‎',0,2)  
findPagesinSubcategories('Category:Entertainment‎',0,2)
findPagesinSubcategories('Category:Fictional_activities‎',0,2)  
findPagesinSubcategories('Category:Fishing‎',0,2)  
findPagesinSubcategories('Category:Food_and_drink_preparation‎',0,2)  
findPagesinSubcategories('Category:Government‎',0,2)
findPagesinSubcategories('Category:Secondary_sector_of_the_economy‎',0,2)  
findPagesinSubcategories('Category:Leisure_activities‎',0,2)  
findPagesinSubcategories('Category:Navigation‎',0,2)  
findPagesinSubcategories('Category:Politics‎',0,2)
findPagesinSubcategories('Category:Planning‎',0,2)  
findPagesinSubcategories('Category:Observation‎',0,2)  
findPagesinSubcategories('Category:Performing_arts‎',0,2)  
findPagesinSubcategories('Category:Physical_exercise‎',0,2)
findPagesinSubcategories('Category:Recreation‎',0,2)  
findPagesinSubcategories('Category:Religion‎',0,2)  
findPagesinSubcategories('Category:Human_spaceflight‎',0,2)  
findPagesinSubcategories('Category:Sports‎',0,2)
findPagesinSubcategories('Category:Transport‎',0,2)  
findPagesinSubcategories('Category:Travel‎',0,2)  
findPagesinSubcategories('Category:Underwater_human_activities‎',0,2)
findPagesinSubcategories('Category:Underwater_diving‎',0,2)  
findPagesinSubcategories('Category:War‎',0,2)  
findPagesinSubcategories('Category:Work‎',0,2)  
findPagesinSubcategories('Category:People‎',0,2)
findPagesinSubcategories('Category:Personal_life‎',0,2)  
findPagesinSubcategories('Category:Self‎',0,2)  
findPagesinSubcategories('Category:Philosophy‎',0,2)  
findPagesinSubcategories('Category:Society‎',0,2)  
findPagesinSubcategories('Category:Social_sciences‎',0,2)
findPagesinSubcategories('Category:Culture‎',0,2)  
findPagesinSubcategories('Category:The_arts‎',0,2)  
findPagesinSubcategories('Category:Research‎',0,2)  
findPagesinSubcategories('Category:Library_science‎',0,2)

import xlsxwriter 

wb = xlsxwriter.Workbook(r'D:\lucru\Licenta Valencia based\Git repo\A-Semi-supervised-Method-to-Classify-Educational-Videos\EnglishWikipediaDataSet-V1.xlsx')
sheet1 = wb.add_worksheet() 
sheet1.write(0, 0, "teest")
n = 0
for list in labelledData:
    sheet1.write( n, 0 ,list[0])
    sheet1.write( n, 1 ,list[1])
    sheet1.write( n, 2 ,list[2])
    n = n + 1

wb.close()
