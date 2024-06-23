#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import threading
import time
from contextlib import ContextDecorator
from functools import wraps


class TimeBenchmark(ContextDecorator):
    """
    Measures execution time using a context manager or decorator.

    This class supports both context manager and decorator usage, and is thread-safe for multithreaded
    environments.

    Args:
        print: If True, prints the elapsed time upon exiting the context or completing the function. Defaults
        to False.

    Usage examples:

        benchmark = TimeBenchmark()

        # As a context manager
        with benchmark:
            time.sleep(1)
        print(f"Block took {benchmark.result:.4f} seconds")

        # As a decorator
        @benchmark
        def example_function():
            time.sleep(1)
            return "Function result"

        result, elapsed_time = example_function()
        print(f"Function took {elapsed_time:.4f} seconds and returned '{result}'")

        # With multithreading
        import threading

        def context_manager_example():
            with benchmark:
                time.sleep(1)
            print(f"Block took {benchmark.result:.4f} seconds")

        def decorator_example():
            result, elapsed_time = example_function()
            print(f"Function took {elapsed_time:.4f} seconds and returned '{result}'")

        threads = []
        for _ in range(3):
            t1 = threading.Thread(target=context_manager_example)
            t2 = threading.Thread(target=decorator_example)
            threads.extend([t1, t2])

        for t in threads:
            t.start()

        for t in threads:
            t.join()
    """

    def __init__(self, print=False):
        self.local = threading.local()
        self.print_time = print

    def __enter__(self):
        self.local.start_time = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.local.end_time = time.perf_counter()
        self.local.elapsed_time = self.local.end_time - self.local.start_time
        if self.print_time:
            print(f"Elapsed time: {self.local.elapsed_time:.4f} seconds")
        return False

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                result = func(*args, **kwargs)
            return result, self.local.elapsed_time

        return wrapper

    @property
    def result(self):
        return getattr(self.local, "elapsed_time", None)

    @property
    def result_ms(self):
        return self.result * 1e3