
import time
import logging

logger = logging.getLogger(__name__)

def retry(func, retries=3, delay=1, backoff=2):
    """Retry a function with exponential backoff.

    Args:
        func: Callable to invoke.
        retries: Maximum number of attempts.
        delay: Initial delay in seconds between attempts.
        backoff: Multiplier applied to delay after each failure.
    """
    current_delay = delay
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if attempt == retries - 1:
                logger.error("All %d attempts failed. Last error: %s", retries, e)
                raise
            logger.warning(
                "Attempt %d/%d failed: %s. Retrying in %.1fs...",
                attempt + 1, retries, e, current_delay,
            )
            time.sleep(current_delay)
            current_delay *= backoff
