PY ?= python
IMAGE ?= heart-risk-api:1.0.0

.PHONY: install data eda train test lint serve mlflow docker-build docker-run k8s-deploy k8s-delete monitoring load

install:      ## create the environment
	$(PY) -m pip install -r requirements.txt

data:         ## download + clean dataset
	$(PY) scripts/download_data.py

eda:          ## EDA figures -> reports/figures
	$(PY) scripts/run_eda.py

train:        ## tune, track in MLflow, package champion -> models/
	$(PY) -m heartrisk.train

lint:
	flake8 src tests scripts

test:
	pytest -v --cov=heartrisk

serve:        ## run API without Docker
	uvicorn heartrisk.api:app --host 0.0.0.0 --port 8000 --reload

mlflow:       ## MLflow UI on http://localhost:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 --name heart-api $(IMAGE)

k8s-deploy:   ## Docker Desktop Kubernetes / minikube
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
	kubectl -n heart-risk rollout status deployment/heart-api
	kubectl -n heart-risk get pods,svc

k8s-delete:
	kubectl delete namespace heart-risk

monitoring:   ## API + Prometheus (9090) + Grafana (3000)
	docker compose up -d --build

load:         ## generate traffic for the dashboards
	$(PY) scripts/load_test.py --n 300
