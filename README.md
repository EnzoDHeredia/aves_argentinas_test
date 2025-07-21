# Bird Classifier API

Este proyecto es una API REST en Flask para clasificar imágenes de aves usando un modelo EfficientNet entrenado en PyTorch.

## ¿Qué hace?
- Recibe una imagen vía POST en `/predict`.
- Devuelve el número de clase predicho y la confianza.

## Estructura principal
- `app.py`: API Flask.
- `bird_classifier.py`: Carga y predicción del modelo.
- `model/modelo_102_B1.pth`: Modelo PyTorch (no se sube al repo).
- `requirements.txt`: Dependencias.
- `Procfile`: Para despliegue en Heroku/Render.

## Uso local
1. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Ejecuta la API:
   ```bash
   python app.py
   ```
3. Haz una petición con una imagen:
   ```bash
   python script.py
   ```

## Despliegue gratuito
Puedes usar plataformas como:
- [Render](https://render.com/): Simple, soporta Flask y archivos grandes.
- [Railway](https://railway.app/): Fácil para apps Python.
- [Hugging Face Spaces](https://huggingface.co/spaces): Para demos ML.
- [Heroku](https://heroku.com/): Requiere buildpacks y archivos pequeños (<500MB).

### Pasos generales
1. Sube tu código a GitHub (sin el modelo pesado si la plataforma lo limita).
2. En la plataforma, configura la variable de entorno `MODEL_PATH` si es necesario.
3. Usa el `Procfile` para indicar el comando de arranque:
   ```
   web: python app.py
   ```
4. Sube el modelo manualmente si la plataforma lo permite (en Render puedes subirlo por SFTP o desde un bucket).

## requirements.txt ejemplo
```
flask
flask-cors
torch
torchvision
pillow
requests
```

## Notas
- No subas el modelo ni imágenes al repo.
- Si necesitas ayuda para subir el modelo a la nube, revisa la documentación de la plataforma elegida.
