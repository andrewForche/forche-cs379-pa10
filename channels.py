"""PA 10: bounded FIFO channels carrying reference-counted heap handles."""

from collections import deque
import threading

from gc_heap import GCHeap


class Channel:
    CLOSED = object()

    def __init__(self, capacity: int, heap: GCHeap) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("Channel capacity must be a positive integer.")
        self._capacity = capacity
        self._heap = heap
        self._buffer = deque()
        self._closed = False
        # The PA9 heap has no internal lock. The driver also uses this lock
        # for heap operations so they cannot race with put's incref.
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)

    def put(self, value: int) -> None:
        """Wait for space, then acquire and buffer one reference to a handle.

        The caller retains its reference, including when a closed put fails.
        """
        with self._not_full:
            while len(self._buffer) >= self._capacity and not self._closed:
                self._not_full.wait()
            if self._closed:
                raise RuntimeError("Cannot put on a closed channel.")
            self._heap.incref(value)
            self._buffer.append(value)
            self._not_empty.notify()

    def get(self):
        """Transfer the oldest buffered reference, or return CLOSED once drained.

        The caller must eventually decref the returned handle.
        """
        with self._not_empty:
            while not self._buffer and not self._closed:
                self._not_empty.wait()
            if not self._buffer:
                return self.CLOSED
            handle = self._buffer.popleft()
            self._not_full.notify()
            return handle

    def close(self) -> None:
        """Wake every waiter and preserve buffered references for draining."""
        with self._not_empty:
            self._closed = True
            self._not_empty.notify_all()
            self._not_full.notify_all()


def producer_consumer_driver(
    heap: GCHeap, values: list, channel: Channel, num_consumers: int = 1
):
    """Allocate and deliver values using one producer and one or more consumers.

    Return (handle, value) pairs in consumer recording order. With multiple
    consumers this order may differ from FIFO dequeue order. Every worker is
    joined, and worker errors are propagated after closing and draining.
    """
    if (not isinstance(num_consumers, int) or isinstance(num_consumers, bool)
            or num_consumers <= 0):
        raise ValueError("The driver needs at least one consumer.")
    if channel._heap is not heap:
        raise ValueError("The driver and channel must share the same heap.")

    delivered = []
    errors = []
    result_lock = threading.Lock()

    def record_error(error):
        with result_lock:
            errors.append(error)
        channel.close()

    def produce():
        try:
            for value in values:
                with channel._lock:
                    handle = heap.allocate(value)
                try:
                    channel.put(handle)
                finally:
                    # Release the creator reference on success, or clean up
                    # our allocation if put failed without taking ownership.
                    with channel._lock:
                        heap.decref(handle)
        except Exception as error:
            record_error(error)
        finally:
            channel.close()

    def consume():
        while True:
            handle = channel.get()
            if handle is Channel.CLOSED:
                return
            try:
                try:
                    with channel._lock:
                        value = heap._objects[handle][0]
                    with result_lock:
                        delivered.append((handle, value))
                finally:
                    with channel._lock:
                        heap.decref(handle)
            except Exception as error:
                # Closing stops production; keep draining owned references.
                record_error(error)

    producer_thread = threading.Thread(target=produce)
    consumer_threads = [threading.Thread(target=consume) for _ in range(num_consumers)]
    for thread in consumer_threads:
        thread.start()
    producer_thread.start()
    producer_thread.join()
    for thread in consumer_threads:
        thread.join()
    if errors:
        raise errors[0]
    return delivered


if __name__ == "__main__":
    heap = GCHeap()
    channel = Channel(capacity=2, heap=heap)
    delivered = producer_consumer_driver(heap, list(range(10)), channel, num_consumers=2)
    print(f"Processed {len(delivered)} values; sum = {sum(v for _, v in delivered)}")
    print(f"Live heap objects after completion: {heap.live_count()}")
