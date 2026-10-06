from locust import HttpUser, between, task


class SentimentUser(HttpUser):

    wait_time = between(1, 2)


    @task
    def predict_sentiment(self):
        self.client.post(
            "/predict",
            json={
                "text": ["المنتج رائع جدًا والخدمة ممتازة"]
            },
        )