"""
Periodic tasks that run directly in the huey consumer process.

A dedicated Huey instance is created for periodic tasks to ensure they
run exactly once, rather than being duplicated across job-type consumers.
"""

import logging

_logger = logging.getLogger(__name__)


def register_periodic_tasks(huey_instance) -> None:
    """
    Register all periodic tasks with the given huey instance.

    Args:
        huey_instance: The huey instance to register tasks with.
    """
    from huey import crontab

    @huey_instance.periodic_task(crontab(minute="*/1"))
    def online_scoring_scheduler():
        """
        Runs every minute to fetch active scorer configs and submit scoring jobs.
        """
        from mlflow.genai.scorers.job import run_online_scoring_scheduler

        try:
            run_online_scoring_scheduler()
        except Exception as e:
            _logger.exception(f"Online scoring scheduler failed: {e!r}")

    _logger.info("Registered online_scoring_scheduler periodic task (runs every 1 minute)")
