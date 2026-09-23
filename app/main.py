import os

from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles

from fastapi.responses import FileResponse

from pydantic import BaseModel, Field

from app.services import (
    predict_ticket,
    similar_tickets
)

from app.database import (
    save_ticket,
    get_tickets
)


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


app = FastAPI(

    title=
        "AI IT Support NLP System",

    description=
        "NLP-based IT support ticket classification and routing",

    version="1.0.0"
)


app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


class TicketRequest(BaseModel):

    title: str = Field(

        min_length=2,

        max_length=255
    )

    description: str = Field(

        min_length=5,

        max_length=5000
    )


class SimilarRequest(BaseModel):

    text: str = Field(

        min_length=5,

        max_length=5000
    )


@app.get("/api/health")
def health():

    return {

        "status": "healthy",

        "service":
            "IT Support NLP"
    }


# ---------------------------------------
# Predict and save ticket
# ---------------------------------------

@app.post("/api/predict")
def predict(
    ticket: TicketRequest
):

    # Run NLP + ML prediction

    prediction = predict_ticket(

        ticket.title,

        ticket.description
    )


    # Save ticket and prediction to PostgreSQL

    ticket_id = save_ticket(

        ticket.title,

        ticket.description,

        prediction
    )


    return {

        "ticket_id":
            ticket_id,

        "ticket":
            ticket.model_dump(),

        "prediction":
            prediction
    }


# ---------------------------------------
# Get saved ticket history
# ---------------------------------------

@app.get("/api/tickets")
def tickets():

    return {

        "tickets":
            get_tickets()
    }


# ---------------------------------------
# Similar tickets
# ---------------------------------------

@app.post("/api/similar")
def similar(
    request: SimilarRequest
):

    results = similar_tickets(

        request.text
    )


    return {

        "results":
            results
    }


# ---------------------------------------
# Frontend
# ---------------------------------------

@app.get("/")
def home():

    return FileResponse(

        os.path.join(

            ROOT,

            "static",

            "index.html"
        )
    )


app.mount(

    "/static",

    StaticFiles(

        directory=os.path.join(

            ROOT,

            "static"
        )
    ),

    name="static"
)