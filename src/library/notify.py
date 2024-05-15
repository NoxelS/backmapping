import requests

from library.config import Keys, config

# Get the ntfy.sh channel from the config
CHANNEL = config(Keys.NTFY_CHANNEL)


def send_basic_notification(message: str):
    """
    Send a basic notification to the ntfy.sh channel

    Args:
        message (str): The message to send
    """
    requests.post(f"https://ntfy.sh/{CHANNEL}", data=message.encode(encoding="utf-8"))


def send_notification(title: str, message: str, tags="warning", priority="urgent") -> bool:
    """
    Send a notification to the ntfy.sh channel

    Args:
        title (str): Title of the message
        message (str): Message to send
        tags (str, optional): Tags for the message. Defaults to "warning".
        priority (str, optional):  Priority of the message. Defaults to "urgent".
    Returns:
        bool: Success status of the request
    """
    res = requests.post(
        f"https://ntfy.sh/{CHANNEL}",
        data=message.encode(encoding="utf-8"),
        headers={"Title": title, "Priority": priority, "Tags": tags},
    )
    return res.status_code == 200
