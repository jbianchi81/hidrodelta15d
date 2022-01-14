# -*- coding: utf-8 -*-

import requests, sqlite3

import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
home = str(Path.home())
import json

config = dict()
proxy_dict = dict()
# from google.colab import drive
# drive.mount('/content/drive')
# ruta = '/content/drive/MyDrive/04_INA/01_HIDRODELTA/'

# Funciones

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

def C_id_corr_guar(id_Mod,est_id):
  ## Carga Simulados
  response = requests.get(
      config["api"]["url"] + '/sim/calibrados/'+str(id_Mod)+'/corridas_guardadas',
      params={'var_id':'2','estacion_id':str(est_id),'includeProno':False},
      headers={'Authorization': 'Bearer ' + config["api"]["token"]},
      proxies=proxy_dict)
  json_response = response.json()
  return json_response

def C_corr_guar(id_Mod,corrida_id,est_id):
  ## Carga Simulados
  response = requests.get(
      config["api"]["url"] + '/sim/calibrados/'+str(id_Mod)+'/corridas_guardadas/'+str(corrida_id),
      params={'var_id':'2','estacion_id':str(est_id),'includeProno':True},
      headers={'Authorization': 'Bearer ' + config["api"]["token"]},
      proxies=proxy_dict)
  json_response = response.json()
  df_sim = pd.DataFrame.from_dict(json_response[0]['series'][0]['pronosticos'],orient='columns')
  df_sim = df_sim.rename(columns={'timestart':'fecha','valor':'h_sim'})
  df_sim = df_sim[['fecha','h_sim']]
  df_sim['fecha'] = pd.to_datetime(df_sim['fecha'])
  df_sim['h_sim'] = df_sim['h_sim'].astype(float)

  df_sim = df_sim.sort_values(by='fecha')
  df_sim.set_index(df_sim['fecha'], inplace=True)
  df_sim.index = df_sim.index.tz_convert(None)#("America/Argentina/Buenos_Aires")
  df_sim.index = df_sim.index - timedelta(hours=3)

  del df_sim['fecha']
  return df_sim

def C_id_corr_ultimas(id_Mod,est_id):
  # Consulta los id de las corridas
  response = requests.get(
      config["api"]["url"] + '/sim/calibrados/'+str(id_Mod)+'/corridas',
      params={'var_id':'2','estacion_id':str(est_id),'includeProno':False},
      headers={'Authorization': 'Bearer ' + config["api"]["token"]},
      proxies=proxy_dict)
  json_res = response.json()
  return json_res

def C_corr_ultimas(id_Mod,corrida_id,est_id):
  response = requests.get(
          config["api"]["url"] + '/sim/calibrados/'+str(id_Mod)+'/corridas/'+str(corrida_id),
          params={'var_id':'2','estacion_id':str(est_id),'includeProno':True},
          headers={'Authorization': 'Bearer ' + config["api"]["token"]},
          proxies=proxy_dict)
  json_response = response.json()
  df_sim = pd.DataFrame.from_dict(json_response['series'][0]['pronosticos'],orient='columns')
  df_sim = df_sim.rename(columns={'timestart':'fecha','valor':'h_sim'})
  df_sim = df_sim[['fecha','h_sim']]
  df_sim['fecha'] = pd.to_datetime(df_sim['fecha'])
  df_sim['h_sim'] = df_sim['h_sim'].astype(float)

  df_sim = df_sim.sort_values(by='fecha')
  df_sim.set_index(df_sim['fecha'], inplace=True)
  df_sim.index = df_sim.index.tz_convert(None)#("America/Argentina/Buenos_Aires")
  df_sim.index = df_sim.index - timedelta(hours=3)

  del df_sim['fecha']
  return df_sim

def ArmaProno(id_Mod,est_id,f_i_prono,f_f_prono):  
  ## Consulta id de las corridas Guardadas
  json_res = C_id_corr_guar(id_Mod,est_id)
  print('Cantidad de corridas Guardada: ',len(json_res))
  # Las guarda en una lista
  lst_corridas = []
  lst_pronoday = [] 
  for corridas in range(len(json_res)):
    lst_corridas.append(json_res[corridas]['cor_id'])
    lst_pronoday.append(json_res[corridas]['forecast_date'])

  # DF de Id y Fecha de las corridas Guardadas
  df_id_cg = pd.DataFrame(lst_corridas, index =lst_pronoday,columns=['cor_id',])
  df_id_cg.index = pd.to_datetime(df_id_cg.index)
  df_id_cg.index = df_id_cg.index.tz_convert(None)
  df_id_cg.index = df_id_cg.index - timedelta(hours=3)
  df_id_cg['Fuente'] = 'Guardada'

  # Filtra las corridas, se queda con las ultimas.
  # Ahora no hace nada porque se estan tomando las ultimas solamente.
  df_id_cg = df_id_cg[df_id_cg.index > f_i_prono].sort_index()

  ## Consulta id de las Ultimas Corridas
  json_res = C_id_corr_ultimas(id_Mod,est_id)
  print('Cantidad de corridas Ultimas: ',len(json_res))
  # Los guarda en una lista
  lst_corridas = []
  lst_pronoday = []
  for corridas in range(len(json_res)):
    lst_corridas.append(json_res[corridas]['cor_id'])
    lst_pronoday.append(json_res[corridas]['forecast_date'])

  # DF de Id y Fecha de las Ultimas corridas 
  df_id_cu = pd.DataFrame(lst_corridas, index =lst_pronoday,columns=['cor_id',])
  df_id_cu.index = pd.to_datetime(df_id_cu.index)
  df_id_cu.index = df_id_cu.index.tz_convert(None)
  df_id_cu.index = df_id_cu.index - timedelta(hours=3)
  df_id_cu['Fuente'] = 'Ultimas'

  df_id = pd.concat([df_id_cg, df_id_cu]).sort_index()
  # Arma un DF vacio que va llenando
  index1H = pd.date_range(start=f_i_prono, end=f_f_prono, freq='1H')
  df_pronos_all = pd.DataFrame(columns=['h_sim','cor_id'],index = index1H)

  # Consulta cada corrida y va actualizando el Df
  for index, row in df_id.T.iteritems():
    idCor = row['cor_id']
    try:
      if row['Fuente'] == 'Guardada':
        df_sim_i = C_corr_guar(id_Mod,idCor,est_id)
      if row['Fuente'] == 'Ultimas':
        df_sim_i = C_corr_ultimas(id_Mod,idCor,est_id)
      
      df_sim_i['cor_id'] = idCor
      df_pronos_all.update(df_sim_i)
    except:
      print('Error en corrida: ',idCor)
    df_pronos_all['h_sim']
  return df_pronos_all

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

def AjustaRegL(df_base):
  ## Modelo
  train = df_base[:].copy()
  var_obj = 'h_obs'
  covariav = ['h_sim',]
  lr = linear_model.LinearRegression()
  X_train = train[covariav]
  Y_train = train[var_obj]
  lr.fit(X_train,Y_train)

  # Create the test features dataset (X_test) which will be used to make the predictions.
  X_test = train[covariav].values 
  # The labels of the model
  Y_test = train[var_obj].values
  Y_predictions = lr.predict(X_test)
  train['Y_predictions'] = Y_predictions

  # The coefficients
  print('Coefficients: \n', lr.coef_)
  # The mean squared error
  mse = mean_squared_error(Y_test, Y_predictions)
  print('Mean squared error: %.5f' % mse)
  # The coefficient of determination: 1 is perfect prediction
  coefDet = r2_score(Y_test, Y_predictions)
  print('r2_score: %.5f' % coefDet)
  train['Error_pred'] =  train['Y_predictions']  - train[var_obj]
  return lr

def run(c):
  global config
  global proxy_dict
  config = c
  if config["use_proxy"]:
      proxy_dict = config["proxy_dict"]
  else:
      proxy_dict = None

  ruta = config["working_dir"] + "/" # '/home/leyden/HIDRODELTA_15D' # 'C:/HIDRODELTA_15D/'

  carpetaFiguras = ruta + 'Figuras/'

  # Conecta BBDD Local
  bbdd_loc = ruta+'BBDDLocal/BD_Delta_14D_01.sqlite'
  connLoc = sqlite3.connect(bbdd_loc)
  cur = connLoc.cursor()

  Id_EstLocal = { 'Parana':29,
                  'SanFernando':52,
                  'NuevaPalmira':1699}

  ahora = datetime.datetime.now()
  mes = str(ahora.month)
  dia = str(ahora.day)
  hora = str(ahora.hour)
  cod = ''.join([mes, dia, hora])
  # Prono

  sql_q = ('''SELECT min(Fecha) as minF FROM (SELECT Id_CB, max(Fecha) as Fecha FROM DataEntrada GROUP BY Id_CB);''')
  df = pd.read_sql(sql_q, connLoc)
  f_fin_obs = pd.to_datetime(df['minF'].values[0]) 
  f_inicio_prono = (f_fin_obs - timedelta(days=60)).replace(hour=0, minute=0, second=0, microsecond=0)
  f_fin_prono = (f_fin_obs + timedelta(days=14))

  id_modelo = 443

  """## Parana"""

  # Carga San Fernando desde la bbdd local
  sql_h_obs = ('''SELECT Fecha, Nivel FROM DataEntrada WHERE Id_CB = 29;''')
  df_Parana_Obs = pd.read_sql(sql_h_obs, connLoc)

  keys =  pd.to_datetime(df_Parana_Obs['Fecha'], format='%Y-%m-%d')
  df_Parana_Obs.set_index(keys, inplace=True)
  del df_Parana_Obs['Fecha']
  df_Parana_Obs.index.rename('fecha', inplace=True)

  # Pronostico. Por ahora solo se repite el ultimo valor los proximos 4 días
  fecha_0 = df_Parana_Obs.index.max() + timedelta(days=1)
  index1dia = pd.date_range(start=fecha_0, end=f_fin_prono, freq='D')
  df_aux_i_prono = pd.DataFrame(index = index1dia,columns=['Nivel','Fecha','Caudal','Id_CB'])
  df_aux_i_prono['Nivel'] = df_Parana_Obs['Nivel'][-1]
  df_aux_i_prono['Fecha'] = df_aux_i_prono.index
  df_aux_i_prono['Caudal'] = np.nan   
  df_aux_i_prono['Id_CB'] = Id_EstLocal['Parana']

  df_aux_i_prono.to_sql('DataEntrada', con = connLoc, if_exists='append',index=False)

  """## San Fernando"""

  estacion_id_SF = 85
  id_obs_SF = 52

  df_SF_sim = ArmaProno(id_modelo,estacion_id_SF,f_inicio_prono,f_fin_prono)
  df_SF_sim = df_SF_sim.dropna()

  # Carga San Fernando desde la bbdd local
  param = [id_obs_SF,]

  sql_h_obs = ('''SELECT Fecha, Nivel as h_obs FROM DataEntrada WHERE Id_CB = ?;''')
  df_SF_Obs = pd.read_sql(sql_h_obs, connLoc,params=param)
  keys =  pd.to_datetime(df_SF_Obs['Fecha'])
  df_SF_Obs.set_index(keys, inplace=True)
  del df_SF_Obs['Fecha']
  df_SF_Obs.index.rename('fecha', inplace=True)

  fig = plt.figure(figsize=(15, 8))
  ax = fig.add_subplot(1, 1, 1)

  ax.plot(df_SF_sim.index, df_SF_sim['h_sim'],label='h_sim')
  ax.plot(df_SF_Obs.index, df_SF_Obs['h_obs'],label='h_obs')

  plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.7)
  plt.tick_params(axis='both', labelsize=16)
  plt.xlabel('Fecha', size=18)
  plt.ylabel('Nivel [m]', size=18)
  plt.legend(prop={'size':16},loc=0)

  f_name = carpetaFiguras+'SF_ObsSim.jpg'
  plt.savefig(f_name, format='jpg')

  #plt.show()
  plt.close()

  """### Modelo"""

  df_ModeloSF = df_SF_sim.join(df_SF_Obs, how = 'inner')
  del df_ModeloSF['cor_id']
  df_ModeloSF = df_ModeloSF.dropna()
  df_ModeloSF.head(2)

  ModeloRL = AjustaRegL(df_ModeloSF)

  # Pronostico
  covariav = ['h_sim',]
  prediccion = ModeloRL.predict(df_SF_sim[covariav].values)
  df_SF_sim['h_pred'] = prediccion

  fig = plt.figure(figsize=(15, 8))
  ax = fig.add_subplot(1, 1, 1)
  ax.plot(df_SF_sim.index, df_SF_sim['h_sim'],'-',color='r',linewidth=1,label='M15D BsAs')
  ax.plot(df_SF_sim.index, df_SF_sim['h_pred'],'-',color='g',linewidth=1,label='Correccion')
  ax.plot(df_SF_Obs.index, df_SF_Obs['h_obs'],'-',color='b',linewidth=1,label='Obs San Fernando')

  plt.xlim(df_SF_sim.index.min(),df_SF_sim.index.max())

  plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.7)
  plt.tick_params(axis='both', labelsize=16)
  plt.xlabel('Fecha', size=18)
  plt.ylabel('Nivel [m]', size=18)
  plt.legend(prop={'size':16},loc=2,ncol=2 )

  f_name = carpetaFiguras+'SF_ObsSimPred.jpg'
  plt.savefig(f_name, format='jpg')

  #plt.show()
  plt.close()

  # Guarda en BBDD Local
  f_guarda = df_SF_Obs.index.max()
  df_sim_guarda = df_SF_sim[df_SF_sim.index>f_guarda].copy()
  df_sim_guarda = df_sim_guarda[['h_pred',]]

  #Pasa a Cero IGN
  df_sim_guarda['h_pred'] = df_sim_guarda['h_pred'] #- 0.53

  df_sim_guarda['emision'] = cod
  df_sim_guarda['Id'] = Id_EstLocal['SanFernando']

  print(df_sim_guarda.head(2))

  df_sim_guarda.to_sql('PronoFrente', con = connLoc, if_exists='replace',index=True, index_label='Fecha')

  """##   Nueva Palmira"""

  estacion_id_NP = 1699
  id_NP_obs = 1699

  df_NP_sim = ArmaProno(id_modelo,estacion_id_NP,f_inicio_prono,f_fin_prono)
  df_NP_sim = df_NP_sim.dropna()

  # Carga NP desde la bbdd local
  param = [id_NP_obs,]
  sql_h_obs = ('''SELECT Fecha, Nivel as h_obs FROM DataEntrada WHERE Id_CB = ?;''')
  df_NP_Obs = pd.read_sql(sql_h_obs, connLoc,params=param)
  keys =  pd.to_datetime(df_NP_Obs['Fecha'])
  df_NP_Obs.set_index(keys, inplace=True)
  del df_NP_Obs['Fecha']
  df_NP_Obs.index.rename('fecha', inplace=True)

  fig = plt.figure(figsize=(15, 8))
  ax = fig.add_subplot(1, 1, 1)

  ax.plot(df_NP_sim.index, df_NP_sim['h_sim'],label='NP_h_sim')
  ax.plot(df_NP_Obs.index, df_NP_Obs['h_obs'],label='NP_h_obs')

  plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.7)
  plt.tick_params(axis='both', labelsize=16)
  plt.xlabel('Fecha', size=18)
  plt.ylabel('Nivel [m]', size=18)
  plt.legend(prop={'size':16},loc=0)
  f_name = carpetaFiguras+'NP_ObsSim.jpg'
  plt.savefig(f_name, format='jpg')

  #plt.show()
  plt.close()

  """### Modelo"""

  df_ModeloNP = df_NP_sim.join(df_NP_Obs, how = 'inner')
  del df_ModeloNP['cor_id']
  df_ModeloNP = df_ModeloNP.dropna()
  df_ModeloNP.head(2)

  ModeloRL_NP = AjustaRegL(df_ModeloNP)

  # Pronostico
  covariav = ['h_sim',]
  prediccion = ModeloRL_NP.predict(df_NP_sim[covariav].values)
  df_NP_sim['h_pred'] = prediccion

  fig = plt.figure(figsize=(15, 8))
  ax = fig.add_subplot(1, 1, 1)
  ax.plot(df_NP_sim.index, df_NP_sim['h_sim'],'-',color='r',linewidth=1,label='M15D NP')
  ax.plot(df_NP_sim.index, df_NP_sim['h_pred'],'-',color='g',linewidth=1,label='Correccion')
  ax.plot(df_NP_Obs.index, df_NP_Obs['h_obs'],'-',color='b',linewidth=1,label='Obs Nueva Palmira')

  plt.xlim(df_SF_sim.index.min(),df_SF_sim.index.max())

  plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.7)
  plt.tick_params(axis='both', labelsize=16)
  plt.xlabel('Fecha', size=18)
  plt.ylabel('Nivel [m]', size=18)
  plt.legend(prop={'size':16},loc=2,ncol=2 )

  f_name = carpetaFiguras+'NP_ObsSimPred.jpg'
  plt.savefig(f_name, format='jpg')

  #plt.show()
  plt.close()

  # Guarda en BBDD Local
  f_guarda = df_NP_Obs.index.max()
  df_sim_guarda = df_NP_sim[df_NP_sim.index>f_guarda].copy()
  df_sim_guarda = df_sim_guarda[['h_pred',]]

  #Pasa a Cero IGN
  df_sim_guarda['h_pred'] = df_sim_guarda['h_pred'] #+ 0.0275

  df_sim_guarda['emision'] = cod
  df_sim_guarda['Id'] = Id_EstLocal['NuevaPalmira']

  print(df_sim_guarda.head(2))

  df_sim_guarda.to_sql('PronoFrente', con = connLoc, if_exists='append',index=True, index_label='Fecha')

  """# Frente

  ## San Fernando
  """

  paramCBF = [Id_EstLocal['SanFernando'],]
  sql_query = ('''SELECT Nivel, Fecha FROM DataEntrada WHERE Id_CB = ?''')
  df_CB_SF = pd.read_sql_query(sql_query, connLoc,params=paramCBF)
  keys =  pd.to_datetime(df_CB_SF['Fecha'])#, format='%Y-%m-%d')								#Convierte a formato fecha la columna [fecha]
  df_CB_SF.set_index(pd.DatetimeIndex(keys), inplace=True)									#Pasa la fecha al indice del dataframe (DatetimeIndex)
  del df_CB_SF['Fecha']																#Elimina el campo fecha que ya es index
  df_CB_SF.index.rename('Fecha', inplace=True)    #Cambia el nombre del indice
  df_CB_SF['SanFernando'] = df_CB_SF['Nivel']  
  del df_CB_SF['Nivel']

  # Prono
  sql_query = ('''SELECT Fecha, h_pred as Nivel FROM PronoFrente WHERE Id = ?''')
  df_CB_SF_Prono = pd.read_sql_query(sql_query, connLoc,params=paramCBF)
  keys =  pd.to_datetime(df_CB_SF_Prono['Fecha'])#, format='%Y-%m-%d')								#Convierte a formato fecha la columna [fecha]
  df_CB_SF_Prono.set_index(pd.DatetimeIndex(keys), inplace=True)									#Pasa la fecha al indice del dataframe (DatetimeIndex)
  del df_CB_SF_Prono['Fecha']																#Elimina el campo fecha que ya es index
  df_CB_SF_Prono.index.rename('Fecha', inplace=True)    #Cambia el nombre del indice
  df_CB_SF_Prono['SanFernando'] = df_CB_SF_Prono['Nivel']  
  del df_CB_SF_Prono['Nivel']

  df_CB_SF =  pd.concat([df_CB_SF, df_CB_SF_Prono], ignore_index=False)

  """## Nueva Palmira"""

  paramCBF = [Id_EstLocal['NuevaPalmira'],]
  sql_query = ('''SELECT Nivel, Fecha FROM DataEntrada WHERE Id_CB = ?''')
  df_CB_NP = pd.read_sql_query(sql_query, connLoc,params=paramCBF)
  keys =  pd.to_datetime(df_CB_NP['Fecha'])#, format='%Y-%m-%d')								#Convierte a formato fecha la columna [fecha]
  df_CB_NP.set_index(pd.DatetimeIndex(keys), inplace=True)									#Pasa la fecha al indice del dataframe (DatetimeIndex)
  del df_CB_NP['Fecha']																#Elimina el campo fecha que ya es index
  df_CB_NP.index.rename('Fecha', inplace=True)    #Cambia el nombre del indice
  df_CB_NP['NuevaPalmira'] = df_CB_NP['Nivel']  
  del df_CB_NP['Nivel']

  # Prono
  sql_query = ('''SELECT Fecha, h_pred as Nivel FROM PronoFrente WHERE Id = ?''')
  df_CB_NP_Prono = pd.read_sql_query(sql_query, connLoc,params=paramCBF)
  keys =  pd.to_datetime(df_CB_NP_Prono['Fecha'])#, format='%Y-%m-%d')								#Convierte a formato fecha la columna [fecha]
  df_CB_NP_Prono.set_index(pd.DatetimeIndex(keys), inplace=True)									#Pasa la fecha al indice del dataframe (DatetimeIndex)
  del df_CB_NP_Prono['Fecha']																#Elimina el campo fecha que ya es index
  df_CB_NP_Prono.index.rename('Fecha', inplace=True)    #Cambia el nombre del indice
  df_CB_NP_Prono['NuevaPalmira'] = df_CB_NP_Prono['Nivel']  
  del df_CB_NP_Prono['Nivel']

  """## Prepara series para Interpolar"""

  df_CB_NP =  pd.concat([df_CB_NP, df_CB_NP_Prono], ignore_index=False)

  indexUnico = pd.date_range(start=df_CB_SF.index.min(), end=df_CB_SF_Prono.index.max(), freq='H')	    # Fechas desde f_inicio a f_fin con un paso de 5 minutos
  df_CB_F = pd.DataFrame(index = indexUnico)

  df_CB_F = df_CB_F.join(df_CB_SF, how = 'left')
  df_CB_F = df_CB_F.join(df_CB_NP, how = 'left')

  del df_CB_SF
  del df_CB_SF_Prono
  del df_CB_NP
  del df_CB_NP_Prono

  df_CB_F['SanFernando'] = df_CB_F['SanFernando'].interpolate(method='linear',limit_direction='backward')
  df_CB_F['NuevaPalmira'] = df_CB_F['NuevaPalmira'].interpolate(method='linear',limit_direction='backward')

  fig = plt.figure(figsize=(15, 8))
  ax = fig.add_subplot(1, 1, 1)

  ax.plot(df_CB_F.index, df_CB_F['SanFernando'],'-',color='r',linewidth=1,label='SanFernando')
  ax.plot(df_CB_F.index, df_CB_F['NuevaPalmira'],'-',color='g',linewidth=1,label='NuevaPalmira')

  plt.axvline(x=ahora,color="black", linestyle="--",linewidth=2)


  plt.xlabel('Fecha')
  plt.ylabel('Altura')
  plt.legend()
  plt.grid()
  plt.title('Condiciones de Borde: Delta Frontal - Resultados')
  f_nameCBF = carpetaFiguras+'04_CBFrente.jpg'
  plt.savefig(f_nameCBF, format='jpg')
  #plt.show()
  plt.close()

  """## Interpola Frente"""

  df_CB_F['aux'] = df_CB_F['SanFernando'] - df_CB_F['NuevaPalmira']

  df_CB_F['Lujan'] = df_CB_F['SanFernando']
  df_CB_F['SanAntonio'] = df_CB_F['SanFernando'] - (df_CB_F['aux']*0.024)
  df_CB_F['CanaldelEste'] = df_CB_F['SanFernando'] - (df_CB_F['aux']*0.077)
  df_CB_F['Palmas'] = df_CB_F['SanFernando'] - (df_CB_F['aux']*0.123)
  df_CB_F['Palmas b'] = df_CB_F['SanFernando'] - (df_CB_F['aux']*0.227)
  df_CB_F['Mini'] = df_CB_F['SanFernando'] - (df_CB_F['aux']*0.388)
  df_CB_F['LaBarquita'] = df_CB_F['SanFernando'] - (df_CB_F['aux']*0.427)
  df_CB_F['BarcaGrande'] = df_CB_F['SanFernando'] - (df_CB_F['aux']*0.493)
  df_CB_F['Correntoso'] = df_CB_F['SanFernando'] - (df_CB_F['aux']*0.598)
  df_CB_F['Guazu'] = df_CB_F['SanFernando'] - (df_CB_F['aux']*0.800)
  df_CB_F['Sauce'] = df_CB_F['SanFernando'] - (df_CB_F['aux']*0.900)
  df_CB_F['Bravo'] = df_CB_F['NuevaPalmira']
  df_CB_F['Gutierrez'] = df_CB_F['NuevaPalmira']

  del df_CB_F['aux']
  df_CB_F['Fecha'] = df_CB_F.index

  df_CB_F2 = pd.melt(df_CB_F, id_vars=['Fecha'], value_vars=['Lujan','SanAntonio','CanaldelEste','Palmas','Palmas b','Mini','LaBarquita','BarcaGrande','Correntoso','Guazu','Sauce','Bravo','Gutierrez'],var_name='Estacion', value_name='Nivel')
  df_CB_F2['Nivel'] = df_CB_F2['Nivel'].round(3)


  df_CB_F2.to_sql('CB_FrenteDelta', con = connLoc, if_exists='replace',index=False)
  connLoc.commit()

  ### Temporal Agrega condBorde Lujan, Gualeguay y Ibicuy
  print('\nTemporal:  ----------------------------------------')
  print('Agrega condBorde Lujan, Gualeguay y Ibicuy: Q cte')
  ## Lujan
  df_aux_i = pd.DataFrame()    
  df_aux_i['Fecha'] = df_CB_F.index
  df_aux_i['Nivel'] = np.nan
  df_aux_i['Caudal'] = 10
  df_aux_i['Id_CB'] = 10  # Lujan
  df_aux_i.to_sql('DataEntrada', con = connLoc, if_exists='append',index=False)

  ## Gualeguay
  df_aux_i = pd.DataFrame()    
  df_aux_i['Fecha'] = df_CB_F.index
  df_aux_i['Nivel'] = np.nan
  df_aux_i['Caudal'] = 10
  df_aux_i['Id_CB'] = 11 # Gualeguay
  df_aux_i.to_sql('DataEntrada', con = connLoc, if_exists='append',index=False)

  ## Ibicuy
  df_aux_i = pd.DataFrame()    
  df_aux_i['Fecha'] = df_CB_F.index
  df_aux_i['Nivel'] = np.nan
  df_aux_i['Caudal'] = 50
  df_aux_i['Id_CB'] = 12 # Ibicuy
  df_aux_i.to_sql('DataEntrada', con = connLoc, if_exists='append',index=False)

  del df_CB_F

if __name__ == "__main__":
    working_dir = "C:/HIDRODELTA_15D"
    with open(working_dir + "/config.json") as f:
        config = json.load(f)
    run(config)
    print("passed run")