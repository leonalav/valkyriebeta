import unittest
import requests
from unittest.mock import patch
from fastapi.testclient import TestClient
from ..api.main import app

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.test_input = {"text": "What is 2+2?"}
        
    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        
    def test_model_inference(self):
        response = self.client.post("/predict", json=self.test_input)
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json())
        
    def test_input_validation(self):
        invalid_input = {"wrong_key": "test"}
        response = self.client.post("/predict", json=invalid_input)
        self.assertEqual(response.status_code, 422)
