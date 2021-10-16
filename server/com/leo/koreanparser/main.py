from fastapi import FastAPI
import logging
import os
import uvicorn
from konlpy.tag import Komoran
from com.leo.koreanparser.subs_db import SubsDb
from com.leo.koreanparser.subs_db import SubsDbEntry

logging.basicConfig(format='%(asctime)s[%(levelname)s] %(message)s', level=logging.DEBUG)

app = FastAPI()

store_path = os.getenv("kosubs.store", default="/store")


def to_view(e: SubsDbEntry):
    return {
        "name": e.video_name,
        "from": e.from_ts,
        "to": e.to_ts
    }


@app.get("/api/ping")
def read_root():
    return {"Hello": "World"}


@app.get("/api/search")
def search(q: str):
    logging.info(f"Requête de recherche : {q}")
    entries = app.subs_db.search(q)
    return [to_view(e) for e in entries]

@app.on_event("startup")
async def startup_event():
    logging.info("Loading Komoran...")
    analyzer = Komoran()
    logging.info("...Komoran loaded")
    subs_db = SubsDb(analyzer)
    subs_db.load(store_path)
    app.subs_db = subs_db

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
