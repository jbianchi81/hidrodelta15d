
# hidrodelta15d

Aplicación para correr el modelo HEC-RAS Delta del Paraná a 15 días. Condición de borde aguas abajo: salida de modelo Delft3D Río de la Plata. 

## Requerimientos:

- HEC-RAS 4.1.0 (https://www.hec.usace.army.mil/software/hec-ras/downloads/HEC-RAS_410_Setup.exe)
- python >=3.8.10
- SO Windows 10 o superior

## Instalación:

	pip install json requests datetime pandas numpy matplotlib seaborn pathlib
Instalar controlador:
Paso 1:

	pip install pyras --upgrade
	pip install pywin32
Paso 2: Run makepy utilities

- Go to the path where Python modules are sitting:
It may look like this -> C:/Users\solo\Anaconda\Lib\site-packages\win32com\client
or C:/Python27\ArcGIS10.2\Lib\site-packages\win32com\client
or C:/Python27\Lib\site-packages\win32com\client
- Open command line at the above (^) path and run $: python makepy.py
select HECRAS River Analysis System (1.1) from the pop-up window
this will build definitions and import modules of RAS-Controller for use

## Configuración:

Edite el archivo config.json y coloque allí los parámetros de conexión a la API (url y token):

	{
	  "api":{
	    "url": "https://alerta.ina.gob.ar/a5",
	    "token": "some_token_value"
	  }
	}

## Uso:

Desde línea de comando: 

	$python Codigos/run.py 
	opciones:     -o,--get_obs True|False    # default=True corre get_obs  
	              -p,--get_prono True|False  # default=True corre get_prono
	              -m,--run_model True|False  # default=True corre autohec
	              -s,--save True|False       # default=True convierte la salida a JSON 
	              -u,--update True|False     # default=True exporta salida a la API de SSIyAH-INA (alerta.ina.gob.ar/a5)  

Desde Python:

	import Codigos.get_obs
	import Codigos.get_prono
	import Codigos.autohec
	import Codigos.salida_upload
	get_obs.run()                 # importa series de datos observados de la API de SSIyAH-INA (alerta.ina.gob.ar/a5) y almacena en base de datos local (BBDDLocal/BD_Delta_14D_01.sqlite)
	get_prono.run()               # importa series de datos simulados de la API de SSIyAH-INA (alerta.ina.gob.ar/a5) y almacena en base de datos local (BBDDLocal/BD_Delta_14D_01.sqlite)
	autohec.run()                 # ejecuta la aplicación HEC-RAS a través de la librería pyras
	salida_upload.run()           # exporta la salida a la API de SSIyAH-INA (alerta.ina.gob.ar/a5)

## Desarrollado por

Instituto Nacional del Agua, Argentina, 2022
