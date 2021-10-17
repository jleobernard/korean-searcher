FROM python:3.9
RUN apt-get update && apt-get upgrade && apt-get install -y default-jre-headless
RUN java -version
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./app /code/app
CMD ["uvicorn", "app.com.leo.koreanparser.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]
