"""
Celery background jobs that power the self-healing loop.

Three recurring tasks:
  1. decay_pass        - runs every 15 min - evaluates confidence decay for all agents
  2. promotion_pass    - runs every 60 min - promotes episodic → semantic
  3. reverify_pass     - runs on-demand   - resolves queued low-confidence nodes

Celery is the default scheduler. Swap to Temporal or Airflow by re-implementing
these task functions - the core logic (DecayEngine, EpisodicPromoter, SelfCorrector)
is backend-agnostic.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any
from celery import Celery
from shama.core.models import DEFAULT_CONFIG
logger = logging.getLogger(__name__)


# Celery app - configure via environment variables in production
def create_celery_app(broker_url: str = "redis://localhost:6379/1") -> Celery:
    """
    Factory that creates the Celery app with SHAMA beat schedule.
    Call this once at startup.

    Usage:
        celery_app = create_celery_app(broker_url="redis://localhost:6379/1")
    """
    app = Celery("shama", broker=broker_url, backend=broker_url)

    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_acks_late=True,              # re-queue on worker crash
        worker_prefetch_multiplier=1,     # fair scheduling
        beat_schedule={
            "shama-decay-pass": {
                "task": "shama.scheduler.tasks.run_decay_pass_all_agents",
                "schedule": DEFAULT_CONFIG.DECAY_CHECK_INTERVAL_MINUTES * 60,  # seconds
            },
            "shama-promotion-pass": {
                "task": "shama.scheduler.tasks.run_promotion_pass_all_agents",
                "schedule": DEFAULT_CONFIG.PROMOTION_CHECK_INTERVAL_MINUTES * 60,
            },
        },
    )

    return app

# Task registry
# Lazy global - set by the user's application at startup via register_shama_context()
_shama_context: dict[str, Any] = {}
celery_app = create_celery_app()

def register_shama_context(context: dict[str, Any]) -> None:
    """
    Inject SHAMA component instances into the task registry.

    Call this once in your application startup BEFORE starting Celery workers.

    Args:
        context: dict with keys:
            - decay_engine: DecayEngine instance
            - promoter: EpisodicPromoter instance
            - corrector: SelfCorrector instance
            - agent_registry: list[str] of agent_ids to process

    Example:
        register_shama_context({
            "decay_engine": decay_engine,
            "promoter": promoter,
            "corrector": corrector,
            "agent_registry": ["agent-001", "agent-002"],
        })
    """
    global _shama_context
    _shama_context = context
    logger.info("SHAMA scheduler context registered with %d agents", len(context.get("agent_registry", [])))

def _run_async(coro) -> Any:
    """Run an async coroutine from a sync Celery task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

# Task: Decay pass - runs every 15 minutes
@celery_app.task(
    name="shama.scheduler.tasks.run_decay_pass_all_agents",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    soft_time_limit=300,
)
def run_decay_pass_all_agents(self) -> dict[str, Any]:
    """
    Run confidence decay pass for every registered agent.
    Auto-deprecates nodes below floor, queues re-verify for nodes below threshold.
    """
    decay_engine = _shama_context.get("decay_engine")
    agent_registry: list[str] = _shama_context.get("agent_registry", [])

    if not decay_engine:
        logger.error("Decay engine not registered. Call register_shama_context() at startup.")
        return {"error": "decay_engine not registered"}

    results = {}
    for agent_id in agent_registry:
        try:
            pass_result = _run_async(decay_engine.run_decay_pass(agent_id))
            results[agent_id] = pass_result.to_dict()
            logger.info("Decay pass complete for agent=%s", agent_id)
        except Exception as exc:
            logger.error("Decay pass failed for agent=%s: %s", agent_id, exc)
            try:
                raise self.retry(exc=exc)
            except self.MaxRetriesExceededError:
                results[agent_id] = {"error": str(exc)}

    return results

# Task: Promotion pass - runs every 60 minutes
@celery_app.task(
    name="shama.scheduler.tasks.run_promotion_pass_all_agents",
    bind=True,
    max_retries=3,
    default_retry_delay=120,
    soft_time_limit=600,
)
def run_promotion_pass_all_agents(self) -> dict[str, Any]:
    """
    Run episodic → semantic promotion for every registered agent.
    """
    promoter = _shama_context.get("promoter")
    agent_registry: list[str] = _shama_context.get("agent_registry", [])

    if not promoter:
        logger.error("Promoter not registered. Call register_shama_context() at startup.")
        return {"error": "promoter not registered"}

    results = {}
    for agent_id in agent_registry:
        try:
            pass_result = _run_async(promoter.run_promotion_pass(agent_id))
            results[agent_id] = pass_result.to_dict()
            logger.info("Promotion pass complete for agent=%s", agent_id)
        except Exception as exc:
            logger.error("Promotion pass failed for agent=%s: %s", agent_id, exc)
            try:
                raise self.retry(exc=exc)
            except self.MaxRetriesExceededError:
                results[agent_id] = {"error": str(exc)}

    return results



# Task: Re-verify specific node - triggered by decay engine
@celery_app.task(
    name="shama.scheduler.tasks.reverify_node",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    soft_time_limit=120,
)
def reverify_node(
    self,
    node_id: str,
    node_type: str,
    agent_id: str,
) -> dict[str, Any]:
    """
    Re-verify a single low-confidence node.
    Enqueued by the decay engine when confidence drops below REVERIFY_THRESHOLD.
    """
    corrector = _shama_context.get("corrector")

    if not corrector:
        logger.error("Corrector not registered.")
        return {"error": "corrector not registered"}

    from uuid import UUID
    try:
        result = _run_async(
            corrector.reverify_node(
                node_id=UUID(node_id),
                node_type=node_type,
                agent_id=agent_id,
            )
        )
        return result.to_dict()
    except Exception as exc:
        logger.error("Re-verify failed for node=%s: %s", node_id, exc)
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            return {"error": str(exc), "node_id": node_id}

# Task: Resolve contradiction - triggered by contradiction detector
@celery_app.task(
    name="shama.scheduler.tasks.resolve_contradiction",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    soft_time_limit=120,
)
def resolve_contradiction(
    self,
    node_a_id: str,
    node_b_id: str,
    entity: str,
    relation: str,
    value_a: str,
    value_b: str,
    llm_winner: str,
    reasoning: str,
) -> dict[str, Any]:
    """
    Resolve a contradiction between two semantic nodes.
    Enqueued immediately when ContradictionDetector finds a conflict.
    """
    corrector = _shama_context.get("corrector")

    if not corrector:
        return {"error": "corrector not registered"}

    from uuid import UUID
    from shama.healing.contradiction import ContradictionResult

    contradiction = ContradictionResult(
        node_a_id=UUID(node_a_id),
        node_b_id=UUID(node_b_id),
        entity=entity,
        relation=relation,
        value_a=value_a,
        value_b=value_b,
        llm_winner=llm_winner,
        reasoning=reasoning,
    )

    try:
        result = _run_async(corrector.resolve_contradiction(contradiction))
        return result.to_dict()
    except Exception as exc:
        logger.error("Contradiction resolution failed: %s", exc)
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            return {"error": str(exc)}