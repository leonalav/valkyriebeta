from locust import HttpUser, task, between, events
import logging
import json
import time
from datetime import datetime
import os
import statistics

class ModelUser(HttpUser):
    # Configurable test parameters
    wait_time = between(1, 3)
    severity_levels = {
        "low": {"users": 10, "spawn_rate": 2, "duration": "1m"},
        "medium": {"users": 50, "spawn_rate": 5, "duration": "5m"},
        "high": {"users": 100, "spawn_rate": 10, "duration": "10m"}
    }
    test_duration = os.getenv("TEST_DURATION", "5m")  # e.g. "10m", "1h"
    ramp_up = os.getenv("RAMP_UP", "30s")  # Time to reach full load
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_config = {
            "endpoint": "/generate",
            "payload": {
                "prompt": "Explain quantum computing in simple terms",
                "max_tokens": 100,
                "temperature": 0.7
            },
            "expected_status": 200,
            "timeout": 30
        }
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

    def setup_logging(self):
        """Configure structured JSON logging and performance thresholds"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler('load_test.json'),
                logging.StreamHandler()
            ]
        )
        
        # Performance thresholds (ms)
        self.performance_thresholds = {
            "warning": 500,
            "critical": 1000,
            "failure_rate": 0.05  # 5% max failure rate
        }
        
        # Test statistics
        self.test_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "start_time": datetime.utcnow().isoformat()
        }
        
        self.logger.info(json.dumps({
            "event": "test_config",
            "config": self.test_config,
            "performance_thresholds": self.performance_thresholds,
            "timestamp": datetime.utcnow().isoformat()
        }))

    @task
    def generate_text(self):
        """Execute load test request with detailed logging and resilience features"""
        start_time = time.time()
        headers = {"Content-Type": "application/json"}
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                with self.client.post(
                    self.test_config["endpoint"],
                    headers=headers,
                    json=self.test_config["payload"],
                    catch_response=True,
                    timeout=self.test_config["timeout"]
                ) as response:
                    response_time = (time.time() - start_time) * 1000  # in ms
                    self._log_response(response, response_time)
                    return response
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    response_time = (time.time() - start_time) * 1000
                    self._log_error(e, response_time)
                    raise
                time.sleep(retry_delay)
                
                response_time = (time.time() - start_time) * 1000  # in ms
                log_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "endpoint": self.test_config["endpoint"],
                    "status_code": response.status_code,
                    "response_time_ms": response_time,
                    "success": response.status_code == self.test_config["expected_status"]
                }

                if response.status_code != self.test_config["expected_status"]:
                    log_data["error"] = f"Status {response.status_code}"
                    response.failure(log_data["error"])
                else:
                    try:
                        result = response.json()
                        if not result.get("text"):
                            log_data["error"] = "Invalid response format"
                            response.failure(log_data["error"])
                    except json.JSONDecodeError:
                        log_data["error"] = "Invalid JSON response"
                        response.failure(log_data["error"])

                self.logger.info(json.dumps(log_data))

            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                self.logger.error(json.dumps({
                    "timestamp": datetime.utcnow().isoformat(),
                    "endpoint": self.test_config["endpoint"],
                    "error": str(e),
                    "response_time_ms": response_time,
                    "success": False
                }))
                raise