from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib

app = FastAPI()

templates = Jinja2Templates(directory="templates")

model = joblib.load("heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
columns = joblib.load("columns.pkl")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    Age: int = Form(...),
    RestingBP: int = Form(...),
    Cholesterol: int = Form(...),
    MaxHR: int = Form(...),
    Oldpeak: float = Form(...)
):

    data = pd.DataFrame({
        "Age": [Age],
        "RestingBP": [RestingBP],
        "Cholesterol": [Cholesterol],
        "MaxHR": [MaxHR],
        "Oldpeak": [Oldpeak]
    })

    data = data.reindex(
        columns=columns,
        fill_value=0
    )

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    result = (
        "Heart Disease Detected"
        if prediction[0] == 1
        else
        "No Heart Disease"
    )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": result
        }
    )
