# src/agent_eval_framework/utils/metric_utils.py
from typing import Any, List, Dict
from vertexai.preview import evaluation
from .logging_utils import log

def _build_metrics(metrics_config: List[Dict[str, Any]]) -> List[str]:
    metrics = []
    if not isinstance(metrics_config, list):
        log.error(f"metrics_config is not a list: {metrics_config}")
        return metrics
    for metric_conf in metrics_config:
        if not isinstance(metric_conf, dict):
            log.warning(f"Skipping invalid metric config: {metric_conf}")
            continue
        metric_name = metric_conf.get("name")
        if not metric_name:
            log.warning(f"Skipping metric with no name: {metric_conf}")
            continue
        metrics.append(metric_name)
        log.info(f"Added metric: {metric_name} (type: {metric_conf.get('type')})")
    return metrics
