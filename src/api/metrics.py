"""
Metrics collection for predictive maintenance API.
Prometheus metrics for monitoring performance and usage.

Author: Brayan Cuevas
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
import psutil
from functools import wraps


# API Metrics
prediction_requests_total = Counter(
    'prediction_requests_total',
    'Total number of prediction requests',
    ['method', 'endpoint', 'status']
)

prediction_request_duration = Histogram(
    'prediction_request_duration_seconds',
    'Time spent processing prediction requests',
    ['method', 'endpoint']
)

model_predictions_total = Counter(
    'model_predictions_total',
    'Total number of model predictions made',
    ['risk_level']
)

api_health_status = Gauge(
    'api_health_status',
    'Health status of the API (1=healthy, 0=unhealthy)'
)

model_loaded_status = Gauge(
    'model_loaded_status',
    'Model loading status (1=loaded, 0=not loaded)'
)

# System Metrics
system_cpu_usage = Gauge(
    'system_cpu_usage_percent',
    'Current CPU usage percentage'
)

system_memory_usage = Gauge(
    'system_memory_usage_percent', 
    'Current memory usage percentage'
)

system_disk_usage = Gauge(
    'system_disk_usage_percent',
    'Current disk usage percentage'
)

# Model Performance Metrics  
model_prediction_confidence = Histogram(
    'model_prediction_confidence',
    'Distribution of model prediction confidence scores'
)

model_response_time = Histogram(
    'model_response_time_seconds',
    'Time taken for model inference'
)


def track_request_metrics(func):
    """Decorator to track request metrics."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        method = "POST"  # Most of our endpoints are POST
        endpoint = func.__name__
        status = "success"
        
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            prediction_request_duration.labels(
                method=method, 
                endpoint=endpoint
            ).observe(duration)
            
            prediction_requests_total.labels(
                method=method,
                endpoint=endpoint, 
                status=status
            ).inc()
    
    return wrapper


def update_system_metrics():
    """Update system resource metrics."""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        system_cpu_usage.set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        system_memory_usage.set(memory.percent)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        system_disk_usage.set(disk_percent)
        
    except Exception as e:
        print(f"Error updating system metrics: {e}")


def update_model_metrics(predictor):
    """Update model-related metrics."""
    try:
        if predictor and predictor.is_ready():
            model_loaded_status.set(1)
            api_health_status.set(1)
        else:
            model_loaded_status.set(0)
            api_health_status.set(0)
    except Exception as e:
        print(f"Error updating model metrics: {e}")
        model_loaded_status.set(0)
        api_health_status.set(0)


def track_prediction_metrics(prediction_result: dict):
    """Track metrics for individual predictions."""
    try:
        # Track by risk level
        risk_level = prediction_result.get('risk_level', 'UNKNOWN')
        model_predictions_total.labels(risk_level=risk_level).inc()
        
        # Track confidence/probability
        confidence = prediction_result.get('failure_probability', 0)
        model_prediction_confidence.observe(confidence)
        
    except Exception as e:
        print(f"Error tracking prediction metrics: {e}")


class MetricsCollector:
    """Centralized metrics collection."""
    
    def __init__(self):
        self.start_time = time.time()
    
    def get_uptime(self):
        """Get service uptime in seconds."""
        return time.time() - self.start_time
    
    def collect_all_metrics(self, predictor=None):
        """Collect all metrics."""
        update_system_metrics()
        update_model_metrics(predictor)
    
    def get_metrics_summary(self):
        """Get summary of key metrics."""
        try:
            return {
                "uptime_seconds": self.get_uptime(),
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "total_predictions": sum([
                    model_predictions_total.labels(risk_level='LOW')._value._value,
                    model_predictions_total.labels(risk_level='MEDIUM')._value._value,
                    model_predictions_total.labels(risk_level='HIGH')._value._value
                ])
            }
        except:
            return {
                "uptime_seconds": self.get_uptime(),
                "cpu_usage": 0,
                "memory_usage": 0, 
                "total_predictions": 0
            }


# Global metrics collector instance
metrics_collector = MetricsCollector()