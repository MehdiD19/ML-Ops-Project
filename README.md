<div align="center">

# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/xhec-mlops-project-student/blob/main/.pre-commit-config.yaml)
</div>

## 👥 Team

This project is developed by:

| Name | GitHub Username | Email |
|------|----------------|-------|
| Mamoun Jamai | [@mamounjamai](https://github.com/mamounjamai) | mamoun.jamai@hec.edu |
| Mehdi Digua | [@MehdiD19](https://github.com/MehdiD19) | mehdi.digua@hec.edu |
| Pierre Lafarguette | [@plafarguette2](https://github.com/plafarguette2) | pierre.lafarguette@hec.edu |
| Marco Salerno | [@sqerbo01](https://github.com/sqerbo01) | marco.salerno@hec.edu |
| Iliass Sijelmassi | [@iliassSjm](https://github.com/iliassSjm) | iliass.sijelmassi@hec.edu |
| Cedric Kire | [@cedrickirek](https://github.com/cedrickirek) | cedric.kire@hec.edu |

---

📥 **Download**: Get the dataset from the [Kaggle page](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)


## 🚀 Quick Start

### Prerequisites
- GitHub account
- [Kaggle account](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2F) (for dataset download)
- Python 3.11

1. **Clone this repository**
2. **Put downloaded dataset in /data folder**
3. **Run ```uv sync``` at the root of the project**
4. **Run ```source .venv/bin/activate``` at the root of the project**

### Run prefect
#### Prerequisites

- Check you have SQLite installed ([Prefect backend database system](https://docs.prefect.io/2.13.7/getting-started/installation/#external-requirements)):
```
sqlite3 --version
```

#### UI setup

- Set an API URL for your local server to make sure that your workflow will be tracked by this specific instance :
```
uv run prefect config set PREFECT_API_URL=http://0.0.0.0:4200/api
```
- Start a local prefect server :
```
uv run prefect server start --host 0.0.0.0
```
- (Optional) If you want to reset the database, run :
```
uv run prefect server database reset
```
- Run the following command in your terminal: 
```
uv run python src/modelling/main.py
```

**Now, you can visit the UI at http://0.0.0.0:4200/dashboard**



### 🌐 Running the Prediction API


#### Local (FastAPI with Uvicorn)

##### Set project root as PYTHONPATH
```export PYTHONPATH=.```

##### Start the FastAPI server
```uv run uvicorn src.web_service.main:app --host 0.0.0.0 --port 8001 --reload```


##### Check Health
```curl -s http://127.0.0.1:8001/health```

##### Make a prediction
```bash
curl -s -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"sex":"M","length":0.455,"diameter":0.365,"height":0.095,"whole_weight":0.514,"shucked_weight":0.2245,"viscera_weight":0.101,"shell_weight":0.15}'
```
#### Docker
##### Build the Docker image
```docker build -f Dockerfile.app -t abalone-api:dev .```

##### Run the container with required port bindings
```docker run --rm -p 0.0.0.0:8000:8001 -p 0.0.0.0:4200:4201 abalone-api:dev```

##### Health check (Docker)
```curl -s http://127.0.0.1:8000/health```

##### Make a prediction (Docker)
```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sex":"M","length":0.455,"diameter":0.365,"height":0.095,"whole_weight":0.514,"shucked_weight":0.2245,"viscera_weight":0.101,"shell_weight":0.15}'
