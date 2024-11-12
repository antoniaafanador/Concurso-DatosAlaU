# Ambiente de desarrollo
Vamos a instalar un ambiente virtual de desarrollo para Python.
Este proyecto asume que usted tiene instalado Python en su computadora y
y que su linea de comandos tiene acceso a Python (variable de entorno).

## Escribir lo siguiente en la linea de comandos
Deberemos abrir el proyecto con la linea de comandos CMD para el caso de Windows,
Bash en el caso de Linux, o Terminal.app en el caso de MacOs)


Luego vamos a la ruta del proyecto y ejecutamos lo siguiente:

## Para Windows CMD
```
python -m venv myenv
myenv\Scripts\activate.bat
python -m pip install --upgrade pip

pip install numpy
pip install pandas
pip install sklearn
pip install scikit-learn
pip install requests
```

## Para Linux y MacOs
```
python3 -m venv myenv
source myenv\bin\activate
python3 -m pip install --upgrade pip

pip install numpy
pip install pandas
pip install sklearn
pip install scikit-learn
pip install requests
```

## Luego podrán ejecutar cualquier script del proyecto.
Ejemplo

Para Linux o MacOs
```
python3 src/prototipo1.py
```

Para Windows
```
python3 src\prototipo1.py
```
