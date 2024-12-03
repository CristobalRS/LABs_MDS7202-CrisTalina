from fastapi import FastAPI
from optimize import optimize_model
# crear aplicación
app = FastAPI()


#Defina GET con ruta tipo home que describa brevemente su modelo, el problema que intenta resolver, su entrada y salida.
@app.get('/') 
async def home(): 
    return {'model': 'XGBoost Optimizado', 'problem': 'Predicción de potabilidad del agua', 'input': 'Características de mediciones de calidad del agua',
        'output': '1 (Potable) o 0 (No potable)', }


#Defina un POST a la ruta /potabilidad/ donde utilice su mejor optimizado para predecir si una medición de agua es o no potable.
@app.post("/potabilidad") 
async def predict(ph: float, Hardness : float, Solids: float, Chloramines:float, Sulfate: float, 
                  Conductivity:float, Organic_carbon : float, Trihalomethanes:float, Turbidity: float)

    model =  optimize_model() 
    prediction = model.predict(ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity)
    result = int(prediction[0])# generar prediccion

    return {"Potabilidad": result} # retornar prediccion
