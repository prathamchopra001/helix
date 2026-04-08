.PHONY: up up-gpu down seed train export-trt test clean logs bootstrap
up:
      docker compose up -d
up-gpu:
      docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
down:
      docker compose down
bootstrap:
      bash scripts/bootstrap.sh
seed:
      bash scripts/seed_data.sh
train:
      docker compose exec training python -m src.trainers.anomaly_trainer
export-trt:
      bash scripts/export_tensorrt.sh
test:
      pytest tests/ -v --cov=services --cov-report=term-missing
logs:
      docker compose logs -f
clean:
      docker compose down -v --remove-orphans