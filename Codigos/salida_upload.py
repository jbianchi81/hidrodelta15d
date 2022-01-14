import pandas as pd
# import math
from datetime import timedelta, datetime
# import os
#from netCDF4 import Dataset
# import numpy as np
import json  
import requests, sqlite3

## globals
config = None
proxy_dict = None
ruta = None

def run(c,upload=True):
    global config
    global proxy_dict
    global ruta
    config = c
    ruta = config["working_dir"] + '/'
    if config["use_proxy"]:
        proxy_dict = config["proxy_dict"]
    else:
        proxy_dict = None

    bbdd_loc = ruta+'BBDDLocal/BD_Delta_14D_01.sqlite'

    conn = sqlite3.connect(bbdd_loc)
    cur = conn.cursor()

    #CHEQUEAR
    #sql_dc = ('''SELECT DiaCorrida FROM SalidasDelta LIMIT 1;''')
    #dia_Corrida = pd.read_sql(sql_dc, conn)
    #reftime = pd.to_datetime(str(dia_Corrida['DiaCorrida'].values[0]))
    reftime = datetime.now()
    forecast_date = reftime.strftime('%Y-%m-%d %H:00:00')
    #forecast_date = datetime(forecast_date.year,forecast_date.month,forecast_date.day,forecast_date.hour)

    cal_id = 454

    Nom_Salidas = 'Estaciones/Dtos_EstDelta.csv'
    archivoSalida = ruta + Nom_Salidas
    lst_output = pd.read_csv(archivoSalida)

    lst_output = lst_output[lst_output['EsSalida']==1]

    # print(lst_output.columns)
    # 'Id', 'X', 'Y', 'River', 'Reach', 'River Stat', 'Node Name', 'NomBBDD',
    # 'NomSalida', 'Est_id', 'cero_escala', 'Series_id_geom','Series_id_hidro'

    list_1 = []
    for index, Salida_i in lst_output.T.iteritems():
        estacion_nombre =   Salida_i['nombre']
        series_id_geom =  Salida_i['S_id_geom']
        series_id_hidro =  Salida_i['S_id_hidro']
        est_id = Salida_i['unid']
        cero = Salida_i['cero_escala']
        
        
        print(estacion_nombre)
        
        param = [est_id,]
        sql_q = ('''SELECT fecha, altura, caudal FROM SalidasDelta WHERE  Id = ?;''')
        data = pd.read_sql(sql_q, conn,params=param)
        data['altura'] = data['altura'].round(2)
        data['caudal'] = data['caudal'].round(2)
        
        
        index_0 = pd.to_datetime(data['fecha'],format='%Y-%m-%d %H:%M:%S')
        data.set_index(index_0, inplace=True)
        del data['fecha']
        data.index = data.index.tz_localize(tz='America/Argentina/Buenos_Aires')
        data.index = data.index.tz_convert(tz="UTC")
        
        data = data.rename(columns={'altura':'valor'})
        
        data['timestart'] = data.index
        data['timeend']= data.index
        data['qualifier']='main'
        
        data = data[['timestart','timeend','valor','qualifier']]
        data['timestart'] = pd.to_datetime(data['timestart'],format='%Y-%m-%d-%H:%M:%S')
        data['timeend'] = pd.to_datetime(data['timeend'],format='%Y-%m-%d-%H:%M:%S')
        
        ## Geom
        result = data.to_json(orient="records",date_format='iso')
        parsed = json.loads(result)
        
        #json.dumps(parsed, indent=4)
        Dic2 = {"series_table":'series',"series_id": series_id_geom,"pronosticos":parsed}
        list_1 = list_1 + [Dic2,]
        
        ## hidrom
        data['valor'] = data['valor'] - cero
        
        result = data.to_json(orient="records",date_format='iso')
        parsed = json.loads(result)
        
        #json.dumps(parsed, indent=4)
        Dic2 = {"series_table":'series',"series_id": series_id_hidro,"pronosticos":parsed}
        list_1 = list_1 + [Dic2,]
        

    Dic1 = {"forecast_date": forecast_date,"cal_id": cal_id,"series":list_1}

    # Serializing json   
    json_object = json.dumps(Dic1, indent = 4)  

    out_file = open(ruta + "SalidasDelta15D.json", "w") 
    json.dump(Dic1, out_file, indent = 4) 
    out_file.close()
    if upload:
        uploadProno(Dic1,cal_id,ruta + 'outputfile.json')

def uploadProno(data,cal_id,responseOutputFile):
    response = requests.post(
        config["api"]["url"] + '/sim/calibrados/' + str(cal_id) + '/corridas',
        data=json.dumps(data),
        headers={'Authorization': 'Bearer ' + config["api"]["token"], 'Content-type': 'application/json'},
        proxies=proxy_dict
    )
    print("prono upload, response code: " + str(response.status_code))
    print("prono upload, reason: " + response.reason)
    if(response.status_code == 200):
        if(responseOutputFile):
            outresponse = open(responseOutputFile,"w")
            outresponse.write(json.dumps(response.json()))
            outresponse.close()

if __name__ == "__main__":
    # carga parametros API
    with open("C:/HIDRODELTA_15D/config.json") as f:
        config = json.load(f)
    run(config)
    print("passed run")