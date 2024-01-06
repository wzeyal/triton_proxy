from locust import HttpUser, task

class InferUser(HttpUser):
    @task
    def infer(self):
        self.client.get("/infer")
        
        
# run locust form this directory
