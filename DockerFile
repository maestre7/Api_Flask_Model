FROM python:3.10-alpine
RUN mkdir /src
WORKDIR /src
ADD . /src
RUN mkdir /src/tmp
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
EXPOSE 5000