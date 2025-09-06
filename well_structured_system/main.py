# Add src to the Python path to allow direct imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from logger import get_logger, set_log_context
import uuid

# --- DEMONSTRATION ---

def process_request(user_id: str):
    """
    A sample function that simulates processing a user request.
    """
    # Get a logger instance for this module
    log = get_logger(__name__)

    log.info("Starting to process user request.")

    try:
        # Some business logic here
        result = 1 / 0
    except Exception as e:
        # The logger will automatically capture the exception info
        log.error(
            "An error occurred during processing.",
            exc_info=True,
            # Pass extra, structured data using the 'extra' keyword
            extra={"details": "Division by zero."}
        )

    log.info(
        "Finished processing request.",
        # More extra data
        extra={"status_code": 200}
    )

def main():
    """
    Main function to demonstrate the logging facade.
    """
    # 1. At the beginning of a session, set the context.
    session_id = str(uuid.uuid4())
    user_id = "user-12345"
    set_log_context(session_id=session_id, user_id=user_id)

    # Get a logger for the main module
    log = get_logger(__name__)
    log.info("Application started.")

    # 2. Call some business logic.
    # The logging context will be automatically propagated.
    process_request(user_id=user_id)

    log.info("Application finished.")


if __name__ == "__main__":
    main()
