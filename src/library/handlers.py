import logging
from library.config import Keys, config
from library.notify import send_notification

def error_handled(throw_exception=False, force_no_ntfy=False):
    """
    Decorator factory to handle errors in functions. This prevents the program from crashing,
    logs the error and sends a notification if the config is set to do so or not set to False.
    
    Args:
        throw_exception (bool, optional): If True, the exception will be raised. Defaults to False.
        use_ntfy (bool, optional): If True, the notification will be sent. Defaults to True.
    
    Returns:
        function: The decorator
        
    Example:
    
    ```python
    from library.handlers import error_handled
    
    @error_handled(throw_exception=True)
    def test_error():
        raise FileNotFoundError("This is a test error")
        
    test_error() # This will log the error, send a notification and raise the exception so the program will crash
    ```
    
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_name = e.__class__.__name__
                logging.error(f"{error_name} in {func.__name__}: {e}")

                if config(Keys.USE_NTFY) and not force_no_ntfy:
                    try:
                        send_notification(
                            title=f"{error_name} in {func.__name__}",
                            message=f"{error_name}: {e}",
                            tags="bangbang",
                        )
                    except Exception as k:
                        logging.error(f"{k.__class__.__name__} in sending notification: {k}")

                if throw_exception:
                    raise e

        return wrapper

    return decorator
