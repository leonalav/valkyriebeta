import json
import logging
import statistics
from typing import Dict, Any

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}',
    handlers=[logging.FileHandler('load_test_results.json')]
)
logger = logging.getLogger(__name__)

def analyze_results(log_file: str) -> Dict[str, Any]:
    """Analyze load test results and return comprehensive metrics"""
    stats = {
        'total_requests': 0,
        'success_rate': 0,
        'response_times': [],
        'failure_reasons': {},
        'endpoints': {},
        'error_severity': {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0
        }
    }

    with open(log_file) as f:
        for line in f:
            try:
                entry = json.loads(line)
                stats['total_requests'] += 1
                
                if entry.get('level') == 'INFO' and 'response_time' in entry.get('message', {}):
                    msg = entry['message']
                    stats['response_times'].append(msg['response_time'])
                    
                    # Track endpoint-specific metrics
                    endpoint = msg.get('endpoint', 'unknown')
                    if endpoint not in stats['endpoints']:
                        stats['endpoints'][endpoint] = {
                            'count': 0,
                            'response_times': []
                        }
                    stats['endpoints'][endpoint]['count'] += 1
                    stats['endpoints'][endpoint]['response_times'].append(msg['response_time'])
                    
                elif entry.get('level') == 'ERROR':
                    msg = entry.get('message', {})
                    reason = msg.get('error')
                    severity = msg.get('severity', 'medium').lower()
                    
                    # Categorize errors
                    stats['failure_reasons'][reason] = stats['failure_reasons'].get(reason, 0) + 1
                    if severity in stats['error_severity']:
                        stats['error_severity'][severity] += 1
                        
            except json.JSONDecodeError:
                continue

    if stats['total_requests'] > 0:
        stats['success_rate'] = len(stats['response_times']) / stats['total_requests']
        
        # Calculate response time metrics
        if stats['response_times']:
            stats['avg_response_time'] = statistics.mean(stats['response_times'])
            stats['p95_response_time'] = statistics.quantiles(stats['response_times'], n=20)[-1]
            stats['p99_response_time'] = statistics.quantiles(stats['response_times'], n=100)[-1]
            stats['max_response_time'] = max(stats['response_times'])
        else:
            stats['avg_response_time'] = 0
            stats['p95_response_time'] = 0
            stats['p99_response_time'] = 0
            stats['max_response_time'] = 0
            
        # Calculate endpoint-specific metrics
        for endpoint, data in stats['endpoints'].items():
            if data['response_times']:
                data['avg_response_time'] = statistics.mean(data['response_times'])
                data['p95_response_time'] = statistics.quantiles(data['response_times'], n=20)[-1]
                data['p99_response_time'] = statistics.quantiles(data['response_times'], n=100)[-1]
            else:
                data['avg_response_time'] = 0
                data['p95_response_time'] = 0
                data['p99_response_time'] = 0

    return stats

def verify_performance(metrics: Dict[str, Any]) -> bool:
    """Verify if performance meets production standards"""
    requirements = {
        'success_rate': 0.99,  # 99% success rate
        'avg_response_time': 2000,  # 2 seconds
        'p95_response_time': 3000  # 3 seconds
    }
    
    logger.info("Performance verification results", extra={
        "metrics": metrics,
        "requirements": requirements
    })
    
    return (metrics['success_rate'] >= requirements['success_rate'] and
            metrics['avg_response_time'] <= requirements['avg_response_time'] and
            metrics['p95_response_time'] <= requirements['p95_response_time'])

if __name__ == "__main__":
    results = analyze_results("load_test.log")
    is_production_ready = verify_performance(results)
    
    if is_production_ready:
        logger.info("Load test passed all performance requirements")
    else:
        logger.error("Load test failed to meet performance requirements")