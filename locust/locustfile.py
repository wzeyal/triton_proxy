from locust import HttpUser, constant_pacing, task

class InferUser(HttpUser):
    
    wait_time=constant_pacing(1)
    
    @task
    def infer(self):
        self.client.get("/infer")
        
        
# run locust form this directory
