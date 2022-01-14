# -*- coding: utf-8 -*-

import sys, getopt
import get_obs
import get_prono
import autohec
import json
import salida_upload

working_dir = "C:/HIDRODELTA_15D"

with open(working_dir + "/config.json") as f:
    config = json.load(f)


# with open(working_dir + "/apiLoginParams.json") as f:
# 	apiLoginParams = json.load(f)

# python39 C:\HIDRODELTA_15D\Codigos\M_hec14dias_1_obs.py
# python39 C:\HIDRODELTA_15D\Codigos\M_hec14dias_2_prono.py
# python39 C:\HIDRODELTA_15D\Codigos\M_hec14dias_3_CorreModelo.py
# python39 C:\HIDRODELTA_15D\Codigos\M_hec14dias_4_aBBDD.py

help_string = 'run.py -h -o bool -p bool -m bool -s bool\
            -o, --get_obs Boolean <True>: Consulta series de datos simulados de la API de a5 y guarda en el archivo SQLite de las entradas del modelo\
            -p, --get_prono Boolean <True>: Consulta series de datos simulados de la API de a5 y guarda en el archivo SQLite de las entradas del modelo\
            -m, --run_model Boolean <True>: a partir de archivo SQLite corre HEC-RAS y genera archivo CSV con las salidas del modelo\
            -s, --save Boolean <True>: convierte archivo CSV de salida de modelo en JSON para POSTear a la API a5\
            -u, --update Boolean <True>: POSTea el JSON de salida del modelo a la API a5'

def main(argv):
    options = {
        "get_obs": True,
        "get_prono": True,
        "run_model": True,
        "save": True,
        "update": True
    }
    try:
        opts, args = getopt.getopt(argv,"ho:p:m:s:u:",["help","get_obs=","get_prono=","run_model=","save=","update="])
    except getopt.GetoptError:
        print(help_string)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_string)
            sys.exit()
        elif opt in ("-o","--get_obs"):
            options["get_obs"] = arg.lower() == 'true' 
        elif opt in ("-p","--get_prono"):
            options["get_prono"] = arg.lower() == 'true' 
        elif opt in ("-m","--run_model"):
            options["run_model"] = arg.lower() == 'true' 
        elif opt in ("-s","--save"):
            options["save"] = arg.lower() == 'true' 
        elif opt in ("-u","--update"):
            options["update"] = arg.lower() == 'true'
    if options["get_obs"]:
        get_obs.run(config)
        print("passed get_obs")
    if options["get_prono"]:
        get_prono.run(config)
        print("passed get_prono")
    if options["run_model"]:
        autohec.run(config)
        print("passed run_model")
    if options["save"]:
        if options["update"]:
            salida_upload.run(config,True)
            print("passed save  & update")
        else:
            salida_upload.run(config,False)
            print("passed save")

if __name__ == "__main__":
   main(sys.argv[1:])
   print("passed run")
