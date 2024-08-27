import logging
import os

from celery import Celery

# Initialize the Celery worker at the module level
redishost = os.getenv("REDISHOST", "localhost")
redisport = int(os.getenv("REDISPORT", 6379))
redisuser = os.getenv("REDISUSER", "")
redispassword = os.getenv("REDISPASSWORD", "")
queue_name = os.getenv("CELERY_QUEUE_NAME", "staging")

# Construct the Redis URL
if redisuser and redispassword:
    redis_url = f"redis://{redisuser}:{redispassword}@{redishost}:{redisport}/0"
else:
    redis_url = f"redis://{redishost}:{redisport}/0"

# Initialize the Celery worker
celery = Celery("KnowledgeGraph", broker=redis_url, backend=redis_url)


# Define queue names
class CeleryWorker:
    def __init__(self, celery_instance):
        self.celery_instance = celery_instance
        self.process_repository_queue = f"{queue_name}_process_repository"
        self.process_file_batch_queue = f"{queue_name}_process_file_batch"
        self.analyze_directory_queue = f"{queue_name}_analyze_directory"
        self.infer_flows_queue = f"{queue_name}_infer_flows"
        logging.info(
            f"Celery worker initialized with queues: {self.process_repository_queue}, {self.process_file_batch_queue}, {self.analyze_directory_queue}, {self.infer_flows_queue}"
        )

    def task_routes(self):
        return {
            "app.modules.parsing.knowledge_graph.code_inference_service.process_repository": {
                "queue": self.process_repository_queue
            },
            "app.modules.parsing.knowledge_graph.code_inference_service.process_file_batch": {
                "queue": self.process_file_batch_queue
            },
            "app.modules.parsing.graph_construction.parsing_service.analyze_directory": {
                "queue": self.analyze_directory_queue
            },
            "app.modules.parsing.knowledge_graph.code_inference_service.infer_flows": {
                "queue": self.infer_flows_queue
            },
        }


# Initialize the celery_worker_instance
celery_worker_instance = CeleryWorker(celery)
