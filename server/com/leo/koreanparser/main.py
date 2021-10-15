from fastapi import FastAPI
import logging
import os
import uvicorn

from com.leo.koreanparser.subs_db import SubsDb

logging.basicConfig(format='%(asctime)s[%(levelname)s] %(message)s', level=logging.DEBUG)

app = FastAPI()
subs_db = SubsDb()

store_path = os.getenv("kosubs.store", default="/store")

@app.get("/api/ping")
def read_root():
    return {"Hello": "World"}

@app.get("/api/search")
def search(q: str):
    logging.info(f"Requête de recherche : {q}")
    return {"q": q}

@app.on_event("startup")
async def startup_event():
    subs_db.load(store_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
