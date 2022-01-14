# -*- coding: utf-8 -*-
### Control de datos de entrada al modelo hidrodinamico ###

### importa librerias
# !pip install psycopg2
import requests # psycopg2, 
import os, sqlite3
import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# ! mkdir Figuras
from pathlib import Path
home = str(Path.home())
import json
# from google.colab import drive
# drive.mount('/content/drive')
# ruta = '/content/drive/MyDrive/04_INA/01_HIDRODELTA/'

config = dict()
proxy_dict = dict()

#FUNCIONES
# Funcion que consulta los observados a la API
def cargaObs(serie_id,timestart,timeend):
  response = requests.get(
      config["api"]["url"] + '/obs/puntual/series/'+str(serie_id)+'/observaciones',
      params={'timestart':timestart,'timeend':timeend},
      headers={'Authorization': 'Bearer ' + config["api"]["token"]},
      proxies=proxy_dict)
  json_response = response.json()

  df_obs_i = pd.DataFrame.from_dict(json_response,orient='columns')
  df_obs_i = df_obs_i[['timestart','valor']]
  df_obs_i = df_obs_i.rename(columns={'timestart':'fecha'})

  df_obs_i['fecha'] = pd.to_datetime(df_obs_i['fecha'])
  df_obs_i['valor'] = df_obs_i['valor'].astype(float)

  df_obs_i = df_obs_i.sort_values(by='fecha')
  df_obs_i.set_index(df_obs_i['fecha'], inplace=True)
  
  df_obs_i.index = df_obs_i.index.tz_convert(None)#("America/Argentina/Buenos_Aires")
  df_obs_i.index = df_obs_i.index - timedelta(hours=3)

  df_obs_i['fecha'] = df_obs_i.index
  df_obs_i = df_obs_i.reset_index(drop=True)
  return df_obs_i

def EliminaSaltos(df_base,umbral_1,umbral_2):
    # Parana
    for index,row in df_base.iterrows():
        if abs(row['Diff_Para']) > umbral_1:
            if (abs(row['Diff_Sant']) > umbral_1) and (abs(row['Diff_Diam']) > umbral_1):
                if abs(row['Diff_Para']) > umbral_2:
                    df_base.loc[index,'Parana']= np.nan
                else:
                    # print('Los 3 presentan un salto. Se supone que esta ok.')
                    continue
            else:
                # print('Salto en Parana y StFe o Diamante')
                df_base.loc[index,'Parana']= np.nan
        else:
            continue

    # Santa Fe
    for index,row in df_base.iterrows():
        if abs(row['Diff_Sant']) > umbral_1:
            if (abs(row['Diff_Para']) > umbral_1) and (abs(row['Diff_Diam']) > umbral_1):
                if abs(row['Diff_Sant']) > umbral_2:
                    df_base.loc[index,'SantaFe']= np.nan
                else:
                    # print('Los 3 presentan un salto. Se supone que esta ok.')
                    continue
            else:
                # print('Salto en StFe y Parana o Diamante')
                df_base.loc[index,'SantaFe']= np.nan
        else:
            continue

    # Diamante
    for index,row in df_base.iterrows():
        if abs(row['Diff_Diam']) > umbral_1:
            if (abs(row['Diff_Para']) > umbral_1) and (abs(row['Diff_Sant']) > umbral_1):
                if abs(row['Diff_Diam']) > umbral_2:
                    df_base.loc[index,'Diamante']= np.nan
                else:
                    # print('Los 3 presentan un salto. Se supone que esta ok.')
                    continue
            else:
                # print('Salto en Diamante y Parana o StFe')
                df_base.loc[index,'Diamante']= np.nan
        else:
            continue
    return df_base

def EliminaSaltos2(df_base,umbral_1,umbral_2):# Elimina Saltos en la serie
    # SFernando
    for index,row in df_base.iterrows():
        if abs(row['Diff_SanF']) > umbral_1:
            if (abs(row['Diff_BsAs']) > umbral_1):
                if abs(row['Diff_SanF']) > umbral_2:
                    df_base.loc[index,'SanFernando']= np.nan
                else:
                    continue
                    #print('Los 3 presentan un salto. Se supone que esta ok.')
            else:
                # print('Salto en SanFer y BsAs o Brag')
                df_base.loc[index,'SanFernando']= np.nan
        else:
            continue
    # BsAs
    for index,row in df_base.iterrows():
        if abs(row['Diff_BsAs']) > umbral_1:
            if (abs(row['Diff_SanF']) > umbral_1):
                if abs(row['Diff_BsAs']) > umbral_2:
                    df_base.loc[index,'BsAs']= np.nan
                else:
                    continue
                    # print('Los 3 presentan un salto. Se supone que esta ok.')
            else:
                # print('Salto en BsAs y SFer o Braga')
                df_base.loc[index,'BsAs']= np.nan
        else:
            continue
    return df_base

def EliminaSaltos3(df_base,umbral_1,umbral_2):# Elimina Saltos en la serie
    # SFernando
    for index,row in df_base.iterrows():
        if abs(row['Diff_Nuev']) > umbral_1:
            if abs(row['Diff_Mart']) > umbral_1:
                if abs(row['Diff_Nuev']) > umbral_2:
                    df_base.loc[index,'Nueva Palmira']= np.nan
                else:
                    continue
                    #print('Los 2 presentan un salto. Se supone que esta ok.')
            else:
                # print('Salto en NPalmira')
                df_base.loc[index,'Nueva Palmira']= np.nan
        else:
            continue
    # BsAs
    for index,row in df_base.iterrows():
        if abs(row['Diff_Mart']) > umbral_1:
            if abs(row['Diff_Nuev']) > umbral_1:
                if abs(row['Diff_Mart']) > umbral_2:
                    df_base.loc[index,'Martinez']= np.nan
                else:
                    continue
                    # print('Los 2 presentan un salto. Se supone que esta ok.')
            else:
                # print('Salto en Martinez')
                df_base.loc[index,'Martinez']= np.nan
        else:
            continue
    return df_base  

# Completa faltantes usando la serie media
def CompletaFaltantes(df_ConMedia):
    ncol = df_ConMedia.columns[0]
    for index,row in df_ConMedia.iterrows():
        if np.isnan(row[ncol]):
            df_ConMedia.loc[index,ncol]= row['medio']
    return df_ConMedia

def run(c):
    global config
    global proxy_dict
    config = c
    if config["use_proxy"]:
        proxy_dict = config["proxy_dict"]
    else:
        proxy_dict = None

    ruta = config["working_dir"] + "/" # '/home/leyden/HIDRODELTA_15D' # 'C:/HIDRODELTA_15D/'

    """# BBDD Local, Estaciones y Fechas"""

    # print('\n ---------------------------------------------------------------------------------------')

    # Conecta BBDD Local
    bbdd_loc = ruta+'BBDDLocal/BD_Delta_14D_01.sqlite'
    connLoc = sqlite3.connect(bbdd_loc)
    cur = connLoc.cursor()

    plotSi = True
    carpetaFiguras = ruta + 'Figuras/'

    Id_EstLocal = { 'Parana':29,
                    'SanFernando':52,
                    'NuevaPalmira':1699}

    # Carga los datos de las estaciones
    Df_Estaciones =  pd.read_csv(ruta + 'Estaciones/Dtos_EstDelta.csv')

    DaysMod = 60          # Largo de la corrida
    ahora = datetime.datetime.now()
    f_fin_0 = (ahora+timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    f_inicio_0 = (f_fin_0 - timedelta(days=DaysMod)).replace(hour=0, minute=0, second=0)

    # Guarda un Id de la corrida tomando Año / mes / dia y hora de la corrida
    nom_clv = str(ahora.year)[2:]+str(ahora.month)+str(ahora.day)+str(ahora.hour)

    print('\n Id Corrida: ',nom_clv)
    print ('Fecha de corrida: Desde: '+f_inicio_0.strftime('%d%b%Y')+ ' Hasta: '+f_fin_0.strftime('%d%b%Y')+'\n')
    Df_Estaciones.head()

    """# Arma CB Observadas

    ## CB Aguas Arriba
    """

    print ('CB Aguas Arriba  ------------------')
    ### Consluta la BBDD

    # Selecciona Estaciones 
    l_idEst = [29,30,31] # CB Aguas Arriba: Parana Santa Fe y Diamante
    Df_EstacionesAA = Df_Estaciones[Df_Estaciones['unid'].isin(l_idEst)]
    print(Df_EstacionesAA[['unid','nombre','series_id']])

    f_inicio_srt = f_inicio_0.strftime("%Y-%m-%d")
    f_fin_srt = f_fin_0.strftime("%Y-%m-%d")

    df_CB_AA = pd.DataFrame(columns=['fecha','valor','id'])
    for index, row in Df_EstacionesAA.T.iteritems():
        id_serie = row['series_id']
        df_obs = cargaObs(id_serie,f_inicio_srt,f_fin_srt)
        print(df_obs.tail(2))
        #print(df_obs['fecha'].max())
        df_obs['id'] = row['unid']
        df_CB_AA = pd.concat([df_CB_AA, df_obs], ignore_index=True)

    del df_obs

    """### Paso 1:

    Une las series en un DF Base:

    Cada serie en una columna. Todas con la misma frecuencia, en este caso diaria.

    También:
    *   Calcula la frecuencia horaria de los datos.
    *   Reemplaza Ceros por NAN.
    *   Calcula diferencias entre valores concecutivos.

    """

    # Crea DF con una frecuencia constante para unir las series
    f_finAA = df_CB_AA['fecha'].max()  # Ahora en lugar de la fecha f_fin_0 toma el maximo de las series consultadas.
    indexUnico = pd.date_range(start=f_inicio_0, end=f_finAA, freq='1D')	    # Fechas desde f_inicio a f_finAA con un paso de 1 Dia
    df_base_CB_AA = pd.DataFrame(index = indexUnico)							      	    # Crea el Df con indexUnico
    df_base_CB_AA.index.rename('fecha', inplace=True)							              
    df_base_CB_AA.index = df_base_CB_AA.index.round("1D")

    df_FrecD = pd.DataFrame()

    for index,row in Df_EstacionesAA.iterrows():
        nombre = (row['nombre'])
        #print(nombre)

        # Toma cada serie del dataframe todo
        df_var = df_CB_AA[(df_CB_AA['id']==row['unid'])].copy()

        # Valores unicos de Horas
        df_var['Horas'] = df_var['fecha'].apply(lambda x: x.hour)
        df_FrecD[nombre] = pd.Series(df_var['Horas'].value_counts())
        # print(df_var['Horas'].value_counts())
        del df_var['Horas']

        #Acomoda DF para unir
        df_var.set_index(pd.DatetimeIndex(df_var['fecha']), inplace=True)   #Pasa la fecha al indice del dataframe (DatetimeIndex)
        del df_var['fecha']
        del df_var['id']
        df_var = df_var.resample('D').mean()
        df_var.columns = [nombre,]
        
        # Une al DF Base
        df_base_CB_AA = df_base_CB_AA.join(df_var, how = 'left')
        
        # Reemplaza Ceros por NAN
        df_base_CB_AA[nombre] = df_base_CB_AA[nombre].replace(0, np.nan)
        
        # Calcula diferencias entre valores concecutivos
        VecDif = np.diff(df_base_CB_AA[nombre].values)
        VecDif = np.append([0,],VecDif)
        coldiff = 'Diff_'+nombre[:4]
        df_base_CB_AA[coldiff] = VecDif

    del df_var
    # print(df_base_CB_AA.head())

    #Frecuencias de los datos por hora
    if plotSi:
        ax = df_FrecD.plot.bar(rot=0)
        plt.title('Frecuencias de los datos por horas del dia')
        plt.tight_layout()
        f_nameFAArriba = carpetaFiguras+'01_1AArriba_Frec.jpg'
        plt.savefig(f_nameFAArriba, format='jpg')
        # plt.show()
        plt.close()


    if plotSi:
        # Reemplaza nan por -1. Son vacíos de la Base.
        df_base_CB_AA[nombre] = df_base_CB_AA[nombre].replace(np.nan,-2)

        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)
        for index,row in Df_EstacionesAA.iterrows():
            nombre = (row['nombre'])
            ax.plot(df_base_CB_AA.index, df_base_CB_AA[nombre],'-',label=nombre)
        plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
        plt.tick_params(axis='both', labelsize=16)
        plt.xlabel('Fecha', size=18)
        plt.ylabel('Nivel [m]', size=18)
        plt.legend(prop={'size':16},loc=0)
        plt.title('AArriba Serie Base')
        plt.tight_layout()
        f_nameS0 = carpetaFiguras+'01_2AArriba_Serie0.jpg'
        plt.savefig(f_nameS0, format='jpg')
        #plt.show()
        plt.close()
        
        # Vuelve a poner los -1 como nan
        df_base_CB_AA[nombre] = df_base_CB_AA[nombre].replace(-2,np.nan)

    df_base_CB_AA

    """### Paso 2:
    Elimina saltos:

    Se establece un umbral_1: si la diferencia entre datos consecutivos supera este umbral_1, se fija si en el resto de las series tambien se produce el salto (se supera el umbral_1).

    Si en todas las series se observa un salto se toma el dato como valido.

    Si el salto no se produce en las tres o si es mayo al segundo umbral_2 (> que el 1ero) se elimina el dato.

    """

    # Datos faltante
    print('\nDatos Faltantes')
    for index,row in Df_EstacionesAA.iterrows():
        nombre = (row['nombre'])
        print('NaN '+nombre+': '+str(df_base_CB_AA[nombre].isna().sum()))

    # print (df_base.head())

    # Elimina Saltos en la serie
    umbral_1 = 0.3
    umbral_2 = 1.0

    # Elimina Saltos
    df_base_CB_AA = EliminaSaltos(df_base_CB_AA,umbral_1,umbral_2)

    print('\nDatos Faltantes Luego de limpiar saltos')
    for index,row in Df_EstacionesAA.iterrows():
        nombre = (row['nombre'])
        print('NaN '+nombre+': '+str(df_base_CB_AA[nombre].isna().sum()))


    if plotSi:
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(df_base_CB_AA.index, df_base_CB_AA['Parana'],'-',label='Parana')
        ax.plot(df_base_CB_AA.index, df_base_CB_AA['SantaFe'],'-',label='SantaFe')
        ax.plot(df_base_CB_AA.index, df_base_CB_AA['Diamante'],'-',label='Diamante')
        plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
        plt.tick_params(axis='both', labelsize=16)
        plt.xlabel('Fecha', size=18)
        plt.ylabel('Nivel [m]', size=18)
        plt.legend(prop={'size':16},loc=0)
        plt.title('CB AArriba')
        plt.tight_layout()
        f_nameS1 = carpetaFiguras+'01_3AArriba_Serie1.jpg'
        plt.savefig(f_nameS1, format='jpg')
        # plt.show()
        plt.close()

    """### Paso 3:
    Completa Faltantes en base a los datos en las otras series.
    1. Lleva las saries al mismo plano.
    2. Calcula meidas de a pares. Parana-Santa Fe , Parana-Diamanate, Parana-Diamante.
    3. Si no hay datos toma el de la media de las otras dos.
    4. Si la diferencia entre el dato y la media es mayor al umbral_3 elimina el dato.
    """

    df_Niveles = df_base_CB_AA.copy()
    df_Niveles = df_Niveles.drop(['Diff_Para', 'Diff_Sant','Diff_Diam'], axis=1)

    # Llevo a la misma face y plano de referencia
    corim_SantaFe = -0.30
    corim_Diamante = -0.30
    df_Niveles['SantaFe'] = df_Niveles['SantaFe'].add(corim_SantaFe)
    df_Niveles['Diamante'] = df_Niveles['Diamante'].add(corim_Diamante)

    # Calcula media de a pares
    df_Niveles['mediaPS'] = df_Niveles[['Parana','SantaFe']].mean(axis = 1,)
    df_Niveles['mediaPD'] = df_Niveles[['Parana','Diamante']].mean(axis = 1,)
    df_Niveles['mediaSD'] = df_Niveles[['SantaFe','Diamante']].mean(axis = 1,)

    print('\nFaltantes de la media de a pares:')
    for mediapar in ['mediaPS','mediaPD','mediaSD']:
        print('NaN '+mediapar+': '+str(df_Niveles[mediapar].isna().sum()))

    # Completa Faltantes
    umbral_3 = 0.3
    for index,row in df_Niveles.iterrows():
        # Parana
        if np.isnan(row['Parana']):
            # print ('Parana Nan')
            df_Niveles.loc[index,'Parana']= row['mediaSD']
        elif    abs(row['Parana']-row['mediaSD']) > umbral_3:
            # print ('Parana Dif Media')
            df_Niveles.loc[index,'Parana']= np.nan
        
        # Santa Fe
        if np.isnan(row['SantaFe']):
            # print ('SantaFe Nan')
            df_Niveles.loc[index,'SantaFe']= row['mediaPD']
        elif    abs(row['SantaFe']-row['mediaPD']) > umbral_3:
            # print ('SantaFe Dif Media')
            df_Niveles.loc[index,'SantaFe']= np.nan

        # Diamante
        if np.isnan(row['Diamante']):
            # print ('Diamante Nan')
            df_Niveles.loc[index,'Diamante']= row['mediaSD']
        elif    abs(row['Diamante']-row['mediaPS']) > umbral_3:
            # print ('Diamante Dif Media')
            df_Niveles.loc[index,'Diamante']= np.nan

    # Faltantes luego de completar con la media de las otras dos series y eliminar cuando hay difrencias mayores a umbral_3
    print('\nFaltentes luego de completar y filtrar:')
    print('NaN Parana: '+str(df_Niveles['Parana'].isna().sum()))
    print('NaN SantaFe: '+str(df_Niveles['SantaFe'].isna().sum()))
    print('NaN Diamante: '+str(df_Niveles['Diamante'].isna().sum()))

    """Interpola de forma Linal"""

    # Interpola para completa todos los fltantes
    df_Niveles = df_Niveles.interpolate(method='linear',limit_direction='backward')
    print('\n Faltentes luego de interpolar:')
    print('NaN Parana: '+str(df_Niveles['Parana'].isna().sum()))
    print('NaN SantaFe: '+str(df_Niveles['SantaFe'].isna().sum()))
    print('NaN Diamante: '+str(df_Niveles['Diamante'].isna().sum()))

    # Vuelve las series a su nivel original
    df_Niveles['SantaFe'] = df_Niveles['SantaFe'].add(-corim_SantaFe)
    df_Niveles['Diamante'] = df_Niveles['Diamante'].add(-corim_Diamante)

    # Series final
    if plotSi:
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(df_Niveles.index, df_Niveles['Parana'],'-',label='Parana')
        ax.plot(df_Niveles.index, df_Niveles['SantaFe'],'-',label='SantaFe')
        ax.plot(df_Niveles.index, df_Niveles['Diamante'],'-',label='Diamante')
        plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
        plt.tick_params(axis='both', labelsize=16)
        plt.xlabel('Fecha', size=18)
        plt.ylabel('Nivel [m]', size=18)
        plt.legend(prop={'size':16},loc=0)
        plt.title('CB Aguas Arriba')
        plt.tight_layout()
        f_nameS2 = carpetaFiguras+'01_4AArriba_Serie2.jpg'
        plt.savefig(f_nameS2, format='jpg')
        #plt.show()
        plt.close()

    df_aux_i = pd.DataFrame()                        # Pasa lista a DF

    # Arma la tabla para guardar en BBDD Local
    cero_parana = Df_Estaciones[Df_Estaciones['nombre']=='Parana']['cero_escala'].values[0]
    df_aux_i['Nivel'] = df_Niveles['Parana'] + cero_parana
    df_aux_i['Fecha'] = df_Niveles.index
    df_aux_i['Caudal'] = np.nan
    df_aux_i['Id_CB'] = Id_EstLocal['Parana']

    # Guarda en la BBDD Local
    df_aux_i.to_sql('DataEntrada', con = connLoc, if_exists='replace',index=False)

    """## CB Frente Margen Derecha"""

    print ('\nCB Frente Margen Derecha  ------------------')
    ### Consluta la BBDD
    l_idE_MDer = [52,85]

    # Margen Derecha
    Df_EstacionesMD = Df_Estaciones[Df_Estaciones['unid'].isin(l_idE_MDer)]
    print(Df_EstacionesMD[['unid','nombre','series_id']])

    f_inicio_srt = f_inicio_0.strftime("%Y-%m-%d")
    f_fin_srt = f_fin_0.strftime("%Y-%m-%d")

    df_CB_MD = pd.DataFrame(columns=['fecha','valor','id'])
    for index, row in Df_EstacionesMD.T.iteritems():
        id_serie = row['series_id']
        df_obs = cargaObs(id_serie,f_inicio_srt,f_fin_srt)
        df_obs['fecha'] = df_obs['fecha'].dt.round('15min') 
    
        df_obs['id'] = row['unid']
        df_CB_MD = pd.concat([df_CB_MD, df_obs], ignore_index=True)
        print(df_obs.tail(2))
    del df_obs

    """### Paso 1:
    Une las series en un DF Base:
    Cada serie en una columna.
    Todas con la misma frecuencia, en este caso diaria.

    También:
    *   Calcula la frecuencia horaria de los datos.
    *   Calcula diferencias entre valores concecutivos.'''
    """

    f_finMD = df_CB_MD['fecha'].max()

    indexUnico15M = pd.date_range(start=f_inicio_0, end=f_finMD, freq='15min')	    #Fechas desde f_inicio a f_fin con un paso de 5 minutos
    df_base_CB_MD = pd.DataFrame(index = indexUnico15M)								#Crea el Df con indexUnico
    df_base_CB_MD.index.rename('fecha', inplace=True)							    #Cambia nombre incide por Fecha

    for index,row in Df_EstacionesMD.iterrows():
        nombre = (row['nombre'])
        # print(nombre)
        df_var = df_CB_MD[(df_CB_MD['id']==row['unid'])].copy()

        #Acomoda DF para unir
        df_var.set_index(pd.DatetimeIndex(df_var['fecha']), inplace=True)   #Pasa la fecha al indice del dataframe (DatetimeIndex)
        del df_var['fecha']    
        del df_var['id']
        df_var.index.round('15min')
        #df_var = df_var.resample('H').mean()
        df_var.columns = [nombre,]
        #print(df_var.tail())
        # Une al DF Base.
        df_base_CB_MD = df_base_CB_MD.join(df_var, how = 'left')
    del df_var

    df_base_CB_MD = df_base_CB_MD.interpolate(method='linear',limit_direction='backward')

    indexUnico1H = pd.date_range(start=f_inicio_0, end=f_finMD, freq='H')	    #Fechas desde f_inicio a f_fin con un paso de 5 minutos
    df_base_CB_MD_H = pd.DataFrame(index = indexUnico1H)								#Crea el Df con indexUnico
    df_base_CB_MD_H.index.rename('fecha', inplace=True)	
    df_base_CB_MD_H = df_base_CB_MD_H.join(df_base_CB_MD, how = 'left')

    df_base_CB_MD = df_base_CB_MD_H.copy()
    del df_base_CB_MD_H

    for index,row in Df_EstacionesMD.iterrows():
        nombre = (row['nombre'])

        # Calcula diferencias entre valores concecutivos
        VecDif = np.diff(df_base_CB_MD[nombre].values)
        VecDif = np.append([0,],VecDif)
        coldiff = 'Diff_'+nombre[:4]
        df_base_CB_MD[coldiff] = VecDif
    #print(df_base_CB_MD.head())


    if plotSi:
        # Reemplaza nan por -2. Son vacíos de la Base
        df_base_CB_MD[nombre] = df_base_CB_MD[nombre].replace(np.nan,-2)

        fig = plt.figure(figsize=(15, 8))
        ax1 = fig.add_subplot(1, 1, 1)
        for index,row in Df_EstacionesMD.iterrows():
            nombre = (row['nombre'])
            ax1.plot(df_base_CB_MD.index, df_base_CB_MD[nombre],'-',label=nombre)

        plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
        plt.tick_params(axis='both', labelsize=16)
        plt.xlabel('Fecha', size=18)
        plt.ylabel('Nivel [m]', size=18)
        plt.legend(prop={'size':16},loc=0)
        plt.title('FMDerecha Serie Base')
        plt.tight_layout()
        f_nameS1 = carpetaFiguras+'02_2FMDerecha_Serie0.jpg'
        plt.savefig(f_nameS1, format='jpg')
        #plt.show()
        plt.close()

        df_base_CB_MD[nombre] = df_base_CB_MD[nombre].replace(-2,np.nan)

    """### Paso 2:
    Elimina saltos:

    Se establece un umbral_1: si la diferencia entre datos consecutivos supera este umbral_1, se fija si en el resto de las series tambien se produce el salto (se supera el umbral_1).

    Si en todas las series se observa un salto se toma el dato como valido.

    Si el salto no se produce en las tres o si es mayo al segundo umbral_2 (> que el 1ero) se elimina el dato.

    """

    # Datos faltante
    print('\nDatos Faltantes')
    for index,row in Df_EstacionesMD.iterrows():
        nombre = (row['nombre'])
        print('NaN '+nombre+': '+str(df_base_CB_MD[nombre].isna().sum()))

    # Elimina Saltos
    umbral_1 = 0.5
    umbral_2 = 1.0

    #  Elimina Saltos
    df_base_CB_MD = EliminaSaltos2(df_base_CB_MD,umbral_1,umbral_2)

    # Datos Faltantes Luego de limpiar saltos
    print('\nDatos Faltantes Luego de limpiar saltos')
    for index,row in Df_EstacionesMD.iterrows():
        nombre = (row['nombre'])
        print('NaN '+nombre+': '+str(df_base_CB_MD[nombre].isna().sum()))

    if plotSi:
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)
        
        ax.plot(df_base_CB_MD.index, df_base_CB_MD['SanFernando'],'-',label='SFernando')
        ax.plot(df_base_CB_MD.index, df_base_CB_MD['BsAs'],'-',label='BsAs')
        plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
        plt.tick_params(axis='both', labelsize=16)
        plt.xlabel('Fecha', size=18)
        plt.ylabel('Nivel [m]', size=18)
        plt.legend(prop={'size':16},loc=0)
        plt.title('Frente Delta Margen Derecha')
        plt.tight_layout()
        f_nameS1 = carpetaFiguras+'02_3FMDerecha_Serie1.jpg'
        plt.savefig(f_nameS1, format='jpg')
        #plt.show()
        plt.close()

    """### Paso 3:
    Completa Faltantes en base a los datos en las otras series.

    1.   Lleva las saries al mismo plano.
    2.   Calcula meidas de a pares. Parana-Santa Fe , Parana-Diamanate, Parana-Diamante.
    3.   Si no hay datos toma el de la media de las otras dos.
    4.   Si la diferencia entre el dato y la media es mayor al umbral_3 elimina el dato.'''
    """

    df_Niveles = df_base_CB_MD.copy()
    df_Niveles = df_Niveles.drop(['Diff_SanF', 'Diff_BsAs'], axis=1)

    # Copia cada serie en un DF distinto
    df_SFer = df_Niveles[['SanFernando']].copy()
    df_BsAs = df_Niveles[['BsAs']].copy()

    # Corrimiento Vertical
    corim_BsAs = 0.2
    df_BsAs['BsAs'] = df_BsAs['BsAs'].add(corim_BsAs)

    # Corrimiento Temporal
    df_BsAs.index = df_BsAs.index + pd.DateOffset(minutes=50)

    # Crea DF para unir todas las series/ frec 5 minutos
    index5m = pd.date_range(start=f_inicio_0, end=f_finMD, freq='5min')	    #Fechas desde f_inicio a f_fin con un paso de 5 minutos
    df_5m = pd.DataFrame(index = index5m)                               #Crea el Df con indexUnico
    df_5m.index.rename('fecha', inplace=True)							              #Cambia nombre incide por Fecha
    df_5m.index = df_5m.index.round("5min")

    # Une en df_5m
    df_5m = df_5m.join(df_SFer, how = 'left')
    df_5m = df_5m.join(df_BsAs, how = 'left')

    # Calcula la media de las tres series. E interpola para completar todos los vacios
    df_5m['medio'] = df_5m[['SanFernando','BsAs',]].mean(axis = 1,)
    df_5m = df_5m.interpolate(method='linear',limit_direction='both')
    #print('\nNaN medio: '+str(df_5m['medio'].isna().sum()))

    # A cada DF de las series les une la serie media de las 3.
    # Siempre se une por Izquierda
    df_SFer = df_SFer.join(df_5m['medio'], how = 'left')
    df_BsAs = df_BsAs.join(df_5m['medio'], how = 'left')

    # print('\n')
    # print('NaN SFernando: '+str(df_SFer['SanFernando'].isna().sum()))
    # print('NaN BsAs: '+str(df_BsAs['BsAs'].isna().sum()))
    # print('NaN Braga: '+str(df_Brag['Braga'].isna().sum()))

    # Completa falrastes usando la serie media
    df_SFer = CompletaFaltantes(df_SFer)
    df_BsAs = CompletaFaltantes(df_BsAs)

    print('\nFaltentes luego de completar con serie media:')
    print('NaN SFernando: '+str(df_SFer['SanFernando'].isna().sum()))
    print('NaN BsAs: '+str(df_BsAs['BsAs'].isna().sum()))

    # Vuelve a llevar las series a su lugar original
    df_BsAs['BsAs'] = df_BsAs['BsAs'].add(-corim_BsAs)
    df_BsAs.index = df_BsAs.index - pd.DateOffset(minutes=50)

    # Une en df_
    df_SFer.columns = ['SanFernando_f','medio']
    df_BsAs.columns = ['BsAs_f','medio']

    df_Niveles = df_Niveles.join(df_SFer['SanFernando_f'], how = 'left')
    df_Niveles = df_Niveles.join(df_BsAs['BsAs_f'], how = 'left')

    del df_SFer
    del df_BsAs

    # Interpola de forma Linal. Maximo 3 dias

    # df_Niveles[nombre] = df_Niveles[nombre].replace(np.nan,-1)
    print('\n Interpola para que no queden faltantes')
    df_Niveles = df_Niveles.interpolate(method='linear',limit_direction='backward')
    print('NaN SFernando: '+str(df_Niveles['SanFernando_f'].isna().sum()))
    print('NaN BsAs: '+str(df_Niveles['BsAs_f'].isna().sum()))

    if plotSi:
        for index,row in Df_EstacionesMD.iterrows():
            nombre = (row['nombre'])
            fig = plt.figure(figsize=(15, 8))
            ax = fig.add_subplot(1, 1, 1)

            nombref = nombre+'_f'
            ax.plot(df_Niveles.index, df_Niveles[nombre],'.',label=nombre)
            # ax.plot(df_Niveles.index, df_Niveles['BsAs'],'.',label='BsAs')
            # ax.plot(df_Niveles.index, df_Niveles['Braga'],'.',label='Braga')
            
            ax.plot(df_Niveles.index, df_Niveles[nombref],'-',label=nombref)
            # ax.plot(df_Niveles.index, df_Niveles['BsAs_f'],'-',label='BsAsF')   
            # ax.plot(df_Niveles.index, df_Niveles['Braga_f'],'-',label='BragaF')
            
            plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
            plt.tick_params(axis='both', labelsize=16)
            plt.xlabel('Fecha', size=18)
            plt.ylabel('Nivel [m]', size=18)
            plt.legend(prop={'size':16},loc=0)
            plt.title('Frente Delta Margen Derecha')
            plt.tight_layout()
            f_nameS1 = carpetaFiguras+'02_4FMDerecha_'+nombre+'.jpg'
            plt.savefig(f_nameS1, format='jpg')
            #plt.show()
            plt.close()

    df_aux_i = pd.DataFrame()                        # Pasa lista a DF
    cero_sanfer = Df_Estaciones[Df_Estaciones['nombre']=='SanFernando']['cero_escala'].values[0]
    df_aux_i['Nivel'] = df_Niveles['SanFernando_f'] + cero_sanfer
    df_aux_i['Fecha'] = df_Niveles.index
    df_aux_i['Id_CB'] = Id_EstLocal['SanFernando']
    df_aux_i.to_sql('DataEntrada', con = connLoc, if_exists='append',index=False)

    del df_Niveles
    del df_aux_i

    """## CB Frente Margen Izquierda"""

    ### Consluta la BBDD
    l_idE_MIzq = [1696,1699]
    # Margen Derecha
    Df_EstacionesMI = Df_Estaciones[Df_Estaciones['unid'].isin(l_idE_MIzq)]
    print(Df_EstacionesMI[['unid','nombre','series_id']])

    f_inicio_srt = f_inicio_0.strftime("%Y-%m-%d")
    f_fin_srt = f_fin_0.strftime("%Y-%m-%d")

    df_CB_MI = pd.DataFrame(columns=['fecha','valor','id'])
    for index, row in Df_EstacionesMI.T.iteritems():
        id_serie = row['series_id']
        df_obs = cargaObs(id_serie,f_inicio_srt,f_fin_srt)
        df_obs['fecha'] = df_obs['fecha'].dt.round('15min') 
        
        df_obs['id'] = row['unid']
        df_CB_MI = pd.concat([df_CB_MI, df_obs], ignore_index=True)

    del df_obs #######################################################################################
    print(df_CB_MI.head())
    print(df_CB_MI.tail())

    """### Paso 1:
    Une las series en un DF Base:

    Cada serie en una columna. Todas con la misma frecuencia, en este caso diaria.

    También:
    *   Calcula la frecuencia horaria de los datos.
    *   Calcula diferencias entre valores concecutivos.
    """

    f_finMI = df_CB_MI['fecha'].max()

    indexUnico15M = pd.date_range(start=f_inicio_0, end=f_finMI, freq='15min')
    df_base_CB_MI = pd.DataFrame(index = indexUnico15M)								#Crea el Df con indexUnico
    df_base_CB_MI.index.rename('fecha', inplace=True)							    #Cambia nombre incide por Fecha

    for index,row in Df_EstacionesMI.iterrows():
        nombre = (row['nombre'])
        #print(nombre)
        df_var = df_CB_MI[(df_CB_MI['id']==row['unid'])].copy()
        
        #Acomoda DF para unir
        df_var.set_index(pd.DatetimeIndex(df_var['fecha']), inplace=True)   #Pasa la fecha al indice del dataframe (DatetimeIndex)
        del df_var['fecha']    
        del df_var['id'] 
        df_var.columns = [nombre,]
        
        # Une al DF Base.
        df_base_CB_MI = df_base_CB_MI.join(df_var, how = 'left')
    del df_var

    #print(df_base_CB_MI.tail(30))
    df_base_CB_MI = df_base_CB_MI.interpolate(method='linear',limit=2,limit_direction='backward')
    #print(df_base_CB_MI.tail(30))

    indexUnico1H = pd.date_range(start=f_inicio_0, end=f_finMD, freq='H')	    #Fechas desde f_inicio a f_fin con un paso de 5 minutos
    df_base_CB_MI_H = pd.DataFrame(index = indexUnico1H)								#Crea el Df con indexUnico
    df_base_CB_MI_H.index.rename('fecha', inplace=True)	
    df_base_CB_MI_H = df_base_CB_MI_H.join(df_base_CB_MI, how = 'left')

    df_base_CB_MI = df_base_CB_MI_H.copy()
    del df_base_CB_MI_H
    #print(df_base_CB_MI.tail(30))
    for index,row in Df_EstacionesMI.iterrows():
        nombre = (row['nombre'])

        # Reemplaza Ceros por NAN
        #df_base[nombre] = df_base[nombre].replace(0, np.nan)

        # Calcula diferencias entre valores concecutivos
        VecDif = abs(np.diff(df_base_CB_MI[nombre].values))
        VecDif = np.append([0,],VecDif)
        coldiff = 'Diff_'+nombre[:4]
        df_base_CB_MI[coldiff] = VecDif


    if plotSi:
        #Reemplaza nan por -1. Son vacíos de la Base
        df_base_CB_MI[nombre] = df_base_CB_MI[nombre].replace(np.nan,-2)

        fig = plt.figure(figsize=(15, 8))
        ax1 = fig.add_subplot(1, 1, 1)

        for index,row in Df_EstacionesMI.iterrows():
            nombre = (row['nombre'])
            ax1.plot(df_base_CB_MI.index, df_base_CB_MI[nombre],'-',label=nombre)
            #coldiff = 'Diff_'+nombre[:4]
        
        plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
        plt.tick_params(axis='both', labelsize=16)
        plt.xlabel('Fecha', size=18)
        plt.ylabel('Nivel [m]', size=18)
        plt.legend(prop={'size':16},loc=0)
        plt.title('FMIzq Serie Base')
        plt.tight_layout()
        f_nameS1 = carpetaFiguras+'03_2FMIzq_Serie0.jpg'
        plt.savefig(f_nameS1, format='jpg')
        #plt.show()
        plt.close()

        df_base_CB_MI[nombre] = df_base_CB_MI[nombre].replace(-2,np.nan)

    """### Paso 2:

    Elimina saltos:

    Se establece un umbral_1: si la diferencia entre datos consecutivos supera este umbral_1, se fija si en el resto de las series tambien se produce el salto (se supera el umbral_1).

    Si en todas las series se observa un salto se toma el dato como valido.

    Si el salto no se produce en las tres o si es mayo al segundo umbral_2 (> que el 1ero) se elimina el dato.
    """

    # Datos faltante
    print('\nDatos Faltantes')
    for index,row in Df_EstacionesMI.iterrows():
        nombre = (row['nombre'])
        print('NaN '+nombre+': '+str(df_base_CB_MI[nombre].isna().sum()))

    # Elimina Saltos
    umbral_1 = 0.3
    umbral_2 = 0.7

    # Elimina Saltos
    df_base_CB_MI = EliminaSaltos3(df_base_CB_MI,umbral_1,umbral_2)


    # Datos Faltantes Luego de limpiar saltos
    print('\nDatos Faltantes Luego de limpiar saltos')

    if plotSi:    
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)

        for index,row in Df_EstacionesMI.iterrows():
            nombre = (row['nombre'])
            #print (nombre)
            print('NaN '+nombre+': '+str(df_base_CB_MI[nombre].isna().sum()))
            ax.plot(df_base_CB_MI.index, df_base_CB_MI[nombre],'-',label=nombre)

        plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
        plt.tick_params(axis='both', labelsize=16)
        plt.xlabel('Fecha', size=18)
        plt.ylabel('Nivel [m]', size=18)
        plt.legend(prop={'size':16},loc=0)
        plt.title('Frente Delta Margen Izquierda')
        plt.tight_layout()
        f_nameS2 = carpetaFiguras+'03_3FMIzq_Serie1.jpg'
        plt.savefig(f_nameS2, format='jpg')
        #plt.show()
        plt.close()

    """### Paso 3:

    Completa Faltantes en base a los datos en las otras series.

    1.   Lleva las saries al mismo plano.
    2.   
    3.   Si no hay datos toma el de la media de las otras dos.
    4.   Si la diferencia entre el dato y la media es mayor al umbral_3 elimina el dato.'''
    """

    # Copia cada serie en un DF distinto
    df_NPal = df_base_CB_MI[['Nueva Palmira']].copy()
    df_Mart = df_base_CB_MI[['Martinez']].copy()

    # Corrimiento Vertical y Temporal
    corim_Mart = 0.24
    df_Mart['Martinez'] = df_Mart['Martinez'].add(corim_Mart)
    df_Mart.index = df_Mart.index - pd.DateOffset(minutes=60)

    # Crea DF para unir todas las series/ frec 1 hora
    index1H = pd.date_range(start=f_inicio_0, end=f_finMI, freq='1H')	    #Fechas desde f_inicio a f_fin con un paso de 5 minutos
    df_1H = pd.DataFrame(index = index1H)								#Crea el Df con indexUnico
    df_1H.index.rename('fecha', inplace=True)							    #Cambia nombre incide por Fecha
    df_1H.index = df_1H.index.round("H")

    # Une en df_1H
    df_1H = df_1H.join(df_NPal, how = 'left')
    df_1H = df_1H.join(df_Mart, how = 'left')

    df_1H['Diff'] = df_1H['Nueva Palmira']-df_1H['Martinez']
    #print(df_1H['Diff'].describe())

    # boxplot = df_1H.boxplot(column=['Diff'])


    df_1H['Nueva Palmira'] = df_1H['Nueva Palmira'].interpolate(method='linear',limit_direction='backward')

    if plotSi:
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)
        
        ax.plot(df_1H.index, df_1H['Nueva Palmira'],'.',label='Nueva Palmira')
        
        plt.grid(True, which='both', color='0.75', linestyle='-.', linewidth=0.5)
        plt.tick_params(axis='both', labelsize=16)
        plt.xlabel('Fecha', size=18)
        plt.ylabel('Nivel [m]', size=18)
        plt.legend(prop={'size':16},loc=0)
        plt.title('Frente Delta Margen Izquierda')
        plt.tight_layout()
        f_nameS1 = carpetaFiguras+'03_4FMIzq.jpg'
        plt.savefig(f_nameS1, format='jpg')
        #plt.show()
        plt.close()

    ################# Guarda para modelo
    df_aux_i = pd.DataFrame()                        # Pasa lista a DF
    cero_NPalmira = Df_Estaciones[Df_Estaciones['nombre']=='Nueva Palmira']['cero_escala'].values[0]
    df_aux_i['Nivel'] = df_1H['Nueva Palmira'] + cero_NPalmira
    df_aux_i['Fecha'] = df_1H.index
    df_aux_i['Id_CB'] = Id_EstLocal['NuevaPalmira']
    df_aux_i = df_aux_i.dropna()
    df_aux_i.to_sql('DataEntrada', con = connLoc, if_exists='append',index=False)

    connLoc.commit()

if __name__ == "__main__":
    # carga parametros API
    with open("C:/HIDRODELTA_15D/config.json") as f:
        config = json.load(f)
    run(config)
    print("passed run")