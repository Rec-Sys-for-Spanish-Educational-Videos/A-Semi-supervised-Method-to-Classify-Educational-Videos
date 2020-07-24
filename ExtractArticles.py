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
                 findPagesinSubcategories(x['title'], depth + 1, label)


 



findPagesinSubcategories('Categoría:Química_orgánica‎',0,1)  
findPagesinSubcategories('Categoría:Química_inorgánica‎',0,1)
findPagesinSubcategories('Categoría:Ingeniería_de_la_edificación‎',0,1)  
findPagesinSubcategories('Categoría:Estadística‎',0,1)  
findPagesinSubcategories('Categoría:Análisis_matemático‎',0,1)  
findPagesinSubcategories('Categoría:Teoría_de_la_demostración‎',0,1)  
findPagesinSubcategories('Categoría:Lógica_matemática‎',0,1)  
findPagesinSubcategories('Categoría:Demostraciones_matemáticas‎',0,1)  
findPagesinSubcategories('Categoría:Teoría_de_números‎',0,1)    
findPagesinSubcategories('Categoría:Informática_teórica‎',0,1)   
findPagesinSubcategories('Categoría:Ingeniería_de_la_computación‎',0,1)   
findPagesinSubcategories('Categoría:Informática‎',0,1)   
findPagesinSubcategories('Categoría:Ingeniería_electrónica‎',0,1)   
findPagesinSubcategories('Categoría:Interfaces_de_programación_de_aplicaciones‎',0,1)   
findPagesinSubcategories('Categoría:Geometría‎',0,1)   

findPagesinSubcategories('Categoría:Gestión_de_proyectos‎',0,2)   
findPagesinSubcategories('Categoría:Finanzas‎',0,2)   
findPagesinSubcategories('Categoría:Lingüística‎',0,2)   
    
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

findPagesinSubcategories('Categoría:Matemáticas‎',0,1)
findPagesinSubcategories('Categoría:Programación lógica‎',0,1)
findPagesinSubcategories('Categoria:Microsoft_Office', 0, 1)
findPagesinSubcategories('Categoria:Ingeniería_de_la_edificación', 0, 1)
findPagesinSubcategories('Categoria:Física', 0, 1)
findPagesinSubcategories('Categoria:Química', 0, 1)
findPagesinSubcategories('Categoria:Ingeniería_agrícola', 0, 1)
findPagesinSubcategories('Categoria:Programación', 0, 1)
findPagesinSubcategories('Categoria:Desarrollo_rural', 0, 1)
findPagesinSubcategories('Categoria:Algoritmos', 0, 1)
findPagesinSubcategories('Categoria:Estructura_de_datos', 0, 1)
findPagesinSubcategories('Categoria:Matemáticas_aplicadas', 0, 1)        


findPagesinSubcategories('Categoría:Arte',0,2)
findPagesinSubcategories('Categoría:Técnicas_de_arte',0,2)
findPagesinSubcategories('Categoría:Antropología',0,2)
findPagesinSubcategories('Categoría:Símbolos',0,2)
findPagesinSubcategories('Categoría:Ciencias_Históricas',0,2)

findPagesinSubcategories('Categoría:Ciencias_sociales',0,2)
findPagesinSubcategories('Categoría:Economía',0,2)
findPagesinSubcategories('Categoría:Sociología',0,2)
findPagesinSubcategories('Categoría:Comunicación',0,2)
findPagesinSubcategories('Categoría:Términos_jurídicos',0,2)
findPagesinSubcategories('Categoría:Justicia',0,2)
findPagesinSubcategories('Categoría:Derecho',0,2)
findPagesinSubcategories('Categoría:Principios_del_derecho',0,2)

findPagesinSubcategories('Categoria:Términos_de_administración', 0, 2) 
findPagesinSubcategories('Categoria:Estrategia', 0, 2)
findPagesinSubcategories('Categoria:Mercadotecnia', 0, 2) 
findPagesinSubcategories('Categoria:Fotografía', 0, 2)
findPagesinSubcategories('Categoria:Administración', 0, 2)
findPagesinSubcategories('Categoria:Ciencias_economico_administrativas', 0, 2)


#wikipedia for tag extracting
findPagesinSubcategories('Categoría:Ciencias_formales', 0, 1) 
findPagesinSubcategories('Categoría:Lingüística_computacional', 0, 1)
findPagesinSubcategories('Categoría:Matemática_financiera', 0, 1)
findPagesinSubcategories('Categoría:Ecuaciones', 0, 1) 
findPagesinSubcategories('Categoría:Ecuaciones_diferenciales', 0, 1)
findPagesinSubcategories('Categoría:Ecuaciones_de_la_física', 0, 1)
findPagesinSubcategories('Categoría:Ecuaciones_epónimas_de_la_física', 0, 1) 
findPagesinSubcategories('Categoría:Heurística', 0, 1)
findPagesinSubcategories('Categoría:Medición', 0, 1)
findPagesinSubcategories('Categoría:Cálculo', 0, 1) 
findPagesinSubcategories('Categoría:Instrumentos_de_medición', 0, 1)
findPagesinSubcategories('Categoría:Topología', 0, 1)
findPagesinSubcategories('Categoría:Teoría_de_la_decisión', 0, 1) 
findPagesinSubcategories('Categoría:Análisis_de_la_varianza', 0, 1)
findPagesinSubcategories('Categoría:Análisis_de_datos', 0, 1)
findPagesinSubcategories('Categoría:Teoría_estadística', 0, 1) 
findPagesinSubcategories('Categoría:Lógica', 0, 1)
findPagesinSubcategories('Categoría:Tecnología', 0, 1)

findPagesinSubcategories('Categoría:Términos_de_biología', 0, 0)
findPagesinSubcategories('Categoría:Teoría_de_la_decisión', 0, 0) 
findPagesinSubcategories('Categoría:Análisis_de_la_varianza', 0, 0)
findPagesinSubcategories('Categoría:Análisis_de_datos', 0, 0)
findPagesinSubcategories('Categoría:Teoría_estadística', 0, 0) 
findPagesinSubcategories('Categoría:Lógica', 0, 0)
findPagesinSubcategories('Categoría:Tecnología', 0, 0)

#extract more 0 and 1 just in case


findPagesinSubcategories('Categoría:Cultura', 0, 2) 
findPagesinSubcategories('Categoría:Arte', 0, 2) 
findPagesinSubcategories('Categoría:Disciplinas_académicas', 0, 2) 
findPagesinSubcategories('Categoría:Artes_liberales', 0, 2) 
findPagesinSubcategories('Categoría:Negocios', 0, 2) 
findPagesinSubcategories('Categoría:Actividades_empresariales', 0, 2) 
findPagesinSubcategories('Categoría:Estudios_empresariales', 0, 2) 
findPagesinSubcategories('Categoría:Inteligencia_empresarial', 0, 2) 
findPagesinSubcategories('Categoría:Edad_Contemporánea', 0, 2) 
findPagesinSubcategories('Categoría:Arte_de_la_Edad_Contemporánea', 0, 2) 
findPagesinSubcategories('Categoría:Arquitectura_contemporánea', 0, 2) 
findPagesinSubcategories('Categoría:Arquitectura_ecléctica', 0, 2) 
findPagesinSubcategories('Categoría:Arquitectura_ecléctica_en_España', 0, 2) 
findPagesinSubcategories('Categoría:Arquitectura_deconstructivista', 0, 2) 
findPagesinSubcategories('Categoría:Estado_socialista', 0, 2) 
findPagesinSubcategories('Categoría:Estados_socialistas', 0, 2) 
findPagesinSubcategories('Categoría:Revolución_bolivariana', 0, 2) 
findPagesinSubcategories('Categoría:Filosofía_contemporánea', 0, 2) 
findPagesinSubcategories('Categoría:Fenomenología', 0, 2) 
findPagesinSubcategories('Categoría:Postestructuralismo', 0, 2) 
findPagesinSubcategories('Categoría:Revolución_Industrial', 0, 2) 
findPagesinSubcategories('Categoría:Multiculturalidad', 0, 2) 
findPagesinSubcategories('Categoría:Interculturalidad', 0, 2) 
findPagesinSubcategories('Categoría:Teoría_de_la_cultura', 0, 2) 
findPagesinSubcategories('Categoría:Historia', 0, 2) 
findPagesinSubcategories('Categoría:Comportamiento_humano', 0, 2) 
findPagesinSubcategories('Categoría:Desarrollo_humano', 0, 2) 
findPagesinSubcategories('Categoría:Ética', 0, 2) 
findPagesinSubcategories('Categoría:Redes_sociales', 0, 2) 
findPagesinSubcategories('Categoría:Colaboración', 0, 2) 
findPagesinSubcategories('Categoría:Industria', 0, 2) 
findPagesinSubcategories('Categoría:Conocimiento', 0, 2) 
findPagesinSubcategories('Categoría:Lingüística_aplicada', 0, 2) 
findPagesinSubcategories('Categoría:Análisis_del_discurso', 0, 2) 
findPagesinSubcategories('Categoría:Pedagogía', 0, 2) 
findPagesinSubcategories('Categoría:Seguridad', 0, 2) 
findPagesinSubcategories('Categoría:Derivados_financieros', 0, 2) 
findPagesinSubcategories('Categoría:Economía_financiera', 0, 2) 
findPagesinSubcategories('Categoría:Finanzas_internacionales', 0, 2) 
findPagesinSubcategories('Categoría:Efectos_económicos', 0, 2) 
findPagesinSubcategories('Categoría:Financiación', 0, 2) 
findPagesinSubcategories('Categoría:Deuda', 0, 2) 
findPagesinSubcategories('Categoría:Hipotecas', 0, 2) 
findPagesinSubcategories('Categoría:Ingreso', 0, 2) 
findPagesinSubcategories('Categoría:Interés', 0, 2) 
findPagesinSubcategories('Categoria:Urbanismo', 0, 2)
findPagesinSubcategories('Categoria:Categoría:Inversión', 0, 2) 
findPagesinSubcategories('Categoría:Activos_financieros', 0, 2) 
findPagesinSubcategories('Categoría:Mercados_de_valores', 0, 2) 
findPagesinSubcategories('Categoría:Mercados_financieros', 0, 2) 
findPagesinSubcategories('Categoría:Bolsas_de_valores', 0, 2) 
findPagesinSubcategories('Categoría:Análisis_bursátiles', 0, 2) 
findPagesinSubcategories('Categoría:Análisis_financiero', 0, 2) 
findPagesinSubcategories('Categoría:Ratios_financieros', 0, 2) 
findPagesinSubcategories('Categoría:Finanzas_públicas', 0, 2) 
findPagesinSubcategories('Categoría:Estado_de_bienestar', 0, 2) 
findPagesinSubcategories('Categoría:Política_fiscal', 0, 2) 
findPagesinSubcategories('Categoría:Derecho_financiero', 0, 2) 
findPagesinSubcategories('Categoría:Riesgo_financiero', 0, 2) 
findPagesinSubcategories('Categoría:Servicios_financieros', 0, 2) 
findPagesinSubcategories('Categoría:Sistemas_de_pago', 0, 2) 
findPagesinSubcategories('Categoría:Globalización', 0, 2) 
findPagesinSubcategories('Categoría:Principios_del_derecho', 0, 2) 
findPagesinSubcategories('Categoría:Términos_jurídicos', 0, 2) 
findPagesinSubcategories('Categoría:Expresiones_latinas_usadas_en_derecho', 0, 2) 
findPagesinSubcategories('Categoría:Política', 0, 2) 
findPagesinSubcategories('Categoría:Terminología_política', 0, 2) 
findPagesinSubcategories('Categoría:Terminología_económica', 0, 2) 
findPagesinSubcategories('Categoría:Terminología_financiera', 0, 2) 
findPagesinSubcategories('Categoría:Inflación', 0, 2) 
findPagesinSubcategories('Categoría:Epónimos_relacionados_con_la_economía', 0, 2) 
findPagesinSubcategories('Categoría:Administración_pública', 0, 2) 
findPagesinSubcategories('Categoría:Seguridad_social', 0, 2) 
findPagesinSubcategories('Categoría:Metodología_de_ciencias_sociales', 0, 2) 
findPagesinSubcategories('Categoría:Teorías_sociológicas', 0, 2) 
findPagesinSubcategories('Categoría:Terminología_sociológica', 0, 2) 




findPagesinSubcategories('Categoría:Ingeniería_de_software', 0, 1) 
findPagesinSubcategories('Categoría:Seguridad_informática', 0, 1) 
findPagesinSubcategories('Categoría:Paradigmas_de_programación', 0, 1) 
findPagesinSubcategories('Categoría:Programación_orientada_a_objetos', 0, 1) 
findPagesinSubcategories('Categoría:Lenguajes_de_programación_orientada_a_objetos', 0, 1) 
findPagesinSubcategories('Categoría:Lenguaje_de_programación_Java', 0, 1) 
findPagesinSubcategories('Categoría:PHP', 0, 1) 
findPagesinSubcategories('Categoría:Python', 0, 1) 
findPagesinSubcategories('Categoría:Software_programado_en_Python', 0, 1) 
findPagesinSubcategories('Categoría:Terminología_informática', 0, 1) 
findPagesinSubcategories('Categoría:Siglas_de_informática', 0, 1) 
findPagesinSubcategories('Categoría:Lenguajes_de_marcado', 0, 1) 
findPagesinSubcategories('Categoría:Lenguajes_de_programación', 0, 1) 
findPagesinSubcategories('Categoría:Teoría_de_tipos', 0, 1) 
findPagesinSubcategories('Categoría:Arquitectura_de_software', 0, 1) 
findPagesinSubcategories('Categoría:Software', 0, 1) 
findPagesinSubcategories('Categoría:Ingeniería_eléctrica', 0, 1) 
findPagesinSubcategories('Categoría:Ingeniería_ambiental', 0, 1) 
findPagesinSubcategories('Categoría:Ingeniería_de_sistemas', 0, 1) 
findPagesinSubcategories('Categoría:Ciencia_de_materiales', 0, 1) 
findPagesinSubcategories('Categoría:Ingeniería_mecánica', 0, 1) 
findPagesinSubcategories('Categoría:Sistemas_operativos_de_tiempo_real', 0, 1) 
findPagesinSubcategories('Categoría:Inteligencia_artificial', 0, 1) 
findPagesinSubcategories('Categoría:Sistemas_operativos', 0, 1) 
findPagesinSubcategories('Categoría:Datos_informáticos', 0, 1) 

findPagesinSubcategories('Categoría:Disciplinas_de_la_biología', 0, 0) 
findPagesinSubcategories('Categoría:Biología_celular', 0, 0) 
findPagesinSubcategories('Categoría:Procesos_celulares', 0, 0) 
findPagesinSubcategories('Categoría:Biología_del_desarrollo', 0, 0) 
findPagesinSubcategories('Categoría:Medicina', 0, 0) 
findPagesinSubcategories('Categoría:Términos_médicos', 0, 0) 

findPagesinSubcategories('Categoría:Zoología', 0, 0) 
findPagesinSubcategories('Categoría:Términos_zoológicos', 0, 0) 

findPagesinSubcategories('Categoría:Tratamientos_en_medicina', 0, 0) 
findPagesinSubcategories('Categoría:Cirugía', 0, 0) 
findPagesinSubcategories('Categoría:Técnicas_quirúrgicas', 0, 0) 
findPagesinSubcategories('Categoría:Anatomía_animal', 0, 0) 
findPagesinSubcategories('Categoría:Anatomía_y_fisiología_de_los_mamíferos', 0, 0) 
findPagesinSubcategories('Categoría:Fisiología_humana', 0, 0) 
findPagesinSubcategories('Categoría:Célula', 0, 0) 
findPagesinSubcategories('Categoría:Tipos_de_células', 0, 0) 
findPagesinSubcategories('Categoría:Nutrición', 0, 0) 
findPagesinSubcategories('Categoría:Enfermedades_metabólicas', 0, 0) 
findPagesinSubcategories('Categoría:Metabolismo', 0, 0) 
findPagesinSubcategories('Categoría:Biogeografía', 0, 0)
findPagesinSubcategories('Categoría:Fisiología_animal', 0, 0) 
findPagesinSubcategories('Categoría:Fisiología_de_los_insectos', 0, 0) 
findPagesinSubcategories('Categoría:Ecología_acuática', 0, 0) 
findPagesinSubcategories('Categoría:Eukaryota', 0, 0) 
findPagesinSubcategories('Categoría:Anatomía_vegetal', 0, 0) 
findPagesinSubcategories('Categoría:Biología_molecular', 0, 0) 
findPagesinSubcategories('Categoría:Biología_marina', 0, 0) 
findPagesinSubcategories('Categoría:Neurociencia', 0, 0) 
findPagesinSubcategories('Categoría:Microbiología', 0, 0) 
findPagesinSubcategories('Categoría:Historia_de_la_medicina', 0, 0) 


import xlsxwriter 

wb = xlsxwriter.Workbook('wikiExtracted.xlsx')
sheet1 = wb.add_worksheet('Sheet 1') 

n = 0
for list in labelledData:
    sheet1.write( n, 0 ,list[0])
    sheet1.write( n, 1 ,list[1])
    sheet1.write( n, 2 ,list[2])
    n = n + 1

wb.close()

