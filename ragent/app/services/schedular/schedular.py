from __future__ import annotations

"""Celery-driven scheduler that continuously looks for executions due in the
next *n* minutes (configurable, default 5), queues them for processing, and
creates the subsequent execution instance based on the agent's schedule.

The scheduler is exposed as a Celery task (``process_scheduled_executions``)
which is executed every minute by Celery Beat.  The logic works as follows:

1. Fetch all ``ExecutionModel`` rows whose ``status`` is ``scheduled`` and the
   ``schedule_time`` is within the *look-ahead* window.
2. For each execution:
   a. Calculate and persist the **next** execution for the same schedule.
   b. Dispatch the *current* execution ID to the workers for processing.
   c. Mark the *current* execution as "running" (effectively *queued*).
3. Commit the transaction and continue.

"""

import os
import calendar
from datetime import datetime, timedelta, timezone
from typing import List, Dict

from app.celery_app import celery_app
from celery import Task
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import (
    ExecutionModel,
    ExecutionStatusEnum,
    ScheduleModel,
    ScheduleTypeEnum,
)
import structlog

__all__: List[str] = [
    "process_scheduled_executions",
]

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _weekday_name_to_int() -> Dict[str, int]:
    """Return a mapping from weekday name (lower-case) to ``datetime.weekday`` int."""
    return {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }


def _compute_next_execution_time(
    schedule: ScheduleModel, last_execution_time: datetime
) -> datetime:
    """Compute the next execution time given a *schedule* and the *last* run.

    This implementation covers *daily*, *weekly* and *monthly* schedules.
    For weekly schedules the next closest day in ``days_of_week`` is chosen.
    For monthly schedules the day is determined by ``day_of_month`` with
    overflow protection (e.g. February 30 -> February 28/29).
    """

    if schedule.schedule_type == ScheduleTypeEnum.daily:  # Every day at the same time
        return last_execution_time + timedelta(days=1)

    if schedule.schedule_type == ScheduleTypeEnum.weekly:
        if not schedule.days_of_week:
            # Fallback: exactly one week later
            return last_execution_time + timedelta(days=7)

        target_weekdays = [_weekday_name_to_int()[d.lower()]
                           for d in schedule.days_of_week]
        # Start searching from the next day
        candidate = last_execution_time + timedelta(days=1)
        while True:
            if candidate.weekday() in target_weekdays:
                return candidate.replace(
                    hour=last_execution_time.hour,
                    minute=last_execution_time.minute,
                    second=last_execution_time.second,
                    microsecond=0,
                )
            candidate += timedelta(days=1)

    if schedule.schedule_type == ScheduleTypeEnum.monthly:
        day = schedule.day_of_month or last_execution_time.day
        # Move to the first day of the *next* month
        year, month = last_execution_time.year, last_execution_time.month + 1
        if month > 12:
            year += 1
            month = 1
        # Cap the day to the last day of that month
        last_day = calendar.monthrange(year, month)[1]
        day = min(day, last_day)
        return datetime(
            year,
            month,
            day,
            last_execution_time.hour,
            last_execution_time.minute,
            last_execution_time.second,
            tzinfo=last_execution_time.tzinfo,
        )

    # Default fallback (should not occur)
    return last_execution_time + timedelta(days=1)


# ---------------------------------------------------------------------------
# Celery tasks
# ---------------------------------------------------------------------------

@celery_app.task(name="app.schedular.schedular.process_scheduled_executions")
def process_scheduled_executions(lookahead_minutes: int | None = None) -> None:  # pragma: no cover
    """Main scheduler task executed every minute by Celery Beat.

    It discovers executions that are due soon, creates their follow-ups, queues
    them for workers, and updates their status to *running* (i.e. *queued*).
    """
    lookahead = lookahead_minutes or int(
        os.getenv("SCHEDULER_LOOKAHEAD_MINUTES", "5"))
    window_end = datetime.now(tz=timezone.utc) + timedelta(minutes=lookahead)

    logger.info(
        "Processing scheduled executions",
        lookahead=lookahead,
        window_end=window_end
    )

    processed_count = 0
    
    with SessionLocal() as session:
        try:
            # 1. Fetch pending executions within the window with row-level locking
            pending_executions: List[ExecutionModel] = (
                session.query(ExecutionModel)
                .filter(
                    ExecutionModel.status == ExecutionStatusEnum.scheduled,
                    ExecutionModel.schedule_time <= window_end,
                )
                .with_for_update(skip_locked=True)  # Skip rows locked by other processes
                .all()
            )

            for execution in pending_executions:
                if execution.schedule is None:
                    logger.warning(
                        "Orphan execution %s - no schedule, skipping", execution.id)
                    continue
                
                try:
                    # 2.a Check if next execution already exists to prevent duplicates
                    next_time = _compute_next_execution_time(
                        execution.schedule, execution.schedule_time)
                    
                    existing_next = session.query(ExecutionModel).filter(
                        ExecutionModel.schedule_id == execution.schedule_id,
                        ExecutionModel.schedule_time == next_time,
                        ExecutionModel.status == ExecutionStatusEnum.scheduled
                    ).first()
                    
                    if not existing_next:
                        # Only create if it doesn't exist
                        new_execution = ExecutionModel(
                            schedule_id=execution.schedule_id,
                            agent_id=execution.agent_id,
                            schedule_time=next_time,
                            status=ExecutionStatusEnum.scheduled,
                        )
                        session.add(new_execution)
                        session.flush()  # Flush to get the ID and catch conflicts early

                    # 2.b Queue current execution for processing
                    logger.info(
                        "Queuing execution for agent processing",
                        execution_id=execution.id,
                        agent_id=execution.agent_id,
                        task_name="app.services.executors.executor.run_agent"
                    )
                    
                    celery_app.send_task(
                        "app.services.executors.executor.run_agent",
                        args=[execution.id, execution.agent_id],
                        queue="default"
                    )
                    
                    logger.info(
                        "Successfully queued execution task",
                        execution_id=execution.id,
                        agent_id=execution.agent_id
                    )

                    # 2.c Mark current execution as *queued*
                    execution.status = ExecutionStatusEnum.queued
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(
                        "Error processing execution %s: %s", 
                        execution.id, str(e)
                    )
                    # Continue with other executions instead of failing the entire batch
                    continue

            # Commit all changes in one transaction
            session.commit()
            
        except Exception as e:
            logger.error("Error in process_scheduled_executions: %s", str(e))
            session.rollback()
            raise

    logger.info("Processed %d scheduled executions", processed_count)
    return processed_count


# ---------------------------------------------------------------------------
# Ensure the scheduler is loaded and periodic task registered
# ---------------------------------------------------------------------------

# Make sure Celery imports this module so that tasks are registered.
# if "app.schedular.schedular" not in celery_app.conf.include:
#     celery_app.conf.include.append("app.schedular.schedular")

# # Register periodic execution (every minute) with Celery Beat *at runtime*.
# celery_app.conf.beat_schedule.update(
#     {
#         "process-scheduled-executions": {
#             "task": "app.schedular.schedular.process_scheduled_executions",
#             "schedule": 60.0,  # every minute
#             "args": (
#                 int(os.getenv("SCHEDULER_LOOKAHEAD_MINUTES", "5")),
#             ),
#         }
#     }
# )
