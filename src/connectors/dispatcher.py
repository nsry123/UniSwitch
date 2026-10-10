"""Non-blocking, isolated delivery workers for configured connectors."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

from src.connectors.base import (
    ActionConnector,
    ActionEvent,
    ConnectorError,
    ConnectorHealth,
    DeliveryResult,
)

ResultCallback = Callable[[DeliveryResult], None]
_STOP = object()


class _ConnectorWorker:
    def __init__(
        self,
        connector: ActionConnector,
        capacity: int,
        result_callback: ResultCallback | None,
    ) -> None:
        self.connector = connector
        self.queue: queue.Queue[ActionEvent | object] = queue.Queue(maxsize=capacity)
        self.result_callback = result_callback
        self.health = ConnectorHealth(False, "not checked")
        self.last_result: DeliveryResult | None = None
        self._lock = threading.Lock()
        self.thread = threading.Thread(
            target=self._run,
            name=f"connector-{connector.connector_id}",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def _record(self, result: DeliveryResult) -> None:
        with self._lock:
            self.last_result = result
        if self.result_callback is not None:
            self.result_callback(result)

    def record_failure(self, event: ActionEvent, message: str) -> None:
        self._record(
            DeliveryResult(
                connector_id=self.connector.connector_id,
                event_id=event.event_id,
                success=False,
                message=message,
                completed_at=datetime.now(UTC),
            )
        )

    def _run(self) -> None:
        try:
            while True:
                item = self.queue.get()
                try:
                    if item is _STOP:
                        return
                    assert isinstance(item, ActionEvent)
                    try:
                        self.connector.send(item)
                    except ConnectorError as error:
                        result = DeliveryResult(
                            self.connector.connector_id,
                            item.event_id,
                            False,
                            str(error),
                            datetime.now(UTC),
                        )
                    except Exception as error:  # keep third-party connectors isolated
                        result = DeliveryResult(
                            self.connector.connector_id,
                            item.event_id,
                            False,
                            f"unexpected connector failure: {type(error).__name__}",
                            datetime.now(UTC),
                        )
                    else:
                        result = DeliveryResult(
                            self.connector.connector_id,
                            item.event_id,
                            True,
                            "event delivered",
                            datetime.now(UTC),
                        )
                    self._record(result)
                finally:
                    self.queue.task_done()
        finally:
            self.connector.close()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result = self.last_result
        return {
            "id": self.connector.connector_id,
            "available": self.health.available,
            "health": self.health.message,
            "queue_depth": self.queue.qsize(),
            "last_delivery": None
            if result is None
            else {
                "event_id": result.event_id,
                "success": result.success,
                "message": result.message,
                "completed_at": result.completed_at.isoformat(),
            },
        }

    def discard_pending(self) -> None:
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                return
            else:
                if isinstance(item, ActionEvent):
                    self.record_failure(item, "event discarded during shutdown")
                self.queue.task_done()


class ConnectorDispatcher:
    """Fan out events without coupling producers to connector latency."""

    def __init__(
        self,
        connectors: Iterable[ActionConnector],
        *,
        queue_capacity: int = 64,
        result_callback: ResultCallback | None = None,
    ) -> None:
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        self._workers = [
            _ConnectorWorker(connector, queue_capacity, result_callback)
            for connector in connectors
        ]
        self._accepting = False
        self._started = False
        self._lock = threading.Lock()

    @property
    def connector_ids(self) -> tuple[str, ...]:
        return tuple(worker.connector.connector_id for worker in self._workers)

    def check_health(self) -> tuple[ConnectorHealth, ...]:
        """Synchronously check connectors before audio capture starts."""
        health: list[ConnectorHealth] = []
        for worker in self._workers:
            worker.health = worker.connector.health_check()
            health.append(worker.health)
        return tuple(health)

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._accepting = True
        for worker in self._workers:
            worker.start()

    def submit(self, event: ActionEvent) -> bool:
        """Queue an event for every connector without waiting for delivery."""
        with self._lock:
            accepting = self._accepting
        if not accepting:
            return False
        accepted = True
        for worker in self._workers:
            try:
                worker.queue.put_nowait(event)
            except queue.Full:
                accepted = False
                worker.record_failure(event, "connector queue is full; event dropped")
        return accepted

    def snapshot(self) -> tuple[dict[str, Any], ...]:
        return tuple(worker.snapshot() for worker in self._workers)

    def close(self, drain_timeout: float = 3.0) -> None:
        """Stop accepting events and drain queued work within a shared timeout."""
        if drain_timeout < 0:
            raise ValueError("drain_timeout must be non-negative")
        with self._lock:
            if not self._started:
                for worker in self._workers:
                    worker.connector.close()
                return
            self._accepting = False
        deadline = time.monotonic() + drain_timeout
        while time.monotonic() < deadline:
            if all(worker.queue.unfinished_tasks == 0 for worker in self._workers):
                break
            time.sleep(0.01)
        for worker in self._workers:
            if worker.queue.unfinished_tasks:
                worker.discard_pending()
            worker.queue.put_nowait(_STOP)
        for worker in self._workers:
            remaining = max(0.0, deadline - time.monotonic())
            worker.thread.join(timeout=remaining)
        with self._lock:
            self._started = False
