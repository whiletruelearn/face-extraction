## Setup 


- Create virtual env (Python 3.9)

```
conda create -n py39 python=3.9
```

- Install dependencies

```
pip install -r requirements.txt
```

- Install youtube-dl package (This is needed to download Age restricted youtube videos without logging in)

```
brew install youtube-dl
```

## Jupyter Notebook

- Run the jupyter notebook `Assignment - ML Engineer*.ipynb` to recreate

## Service 

Bring up the fast API service 

```
docker build -t fastapi-service .
docker run -p 8000:8000 fastapi-service
```

Go to localhost:8000/docs to try out service.