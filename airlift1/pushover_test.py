import requests

PUSHOVER_API_URL = "https://api.pushover.net/1/messages.json"

TOKEN = "aem37vgi2ahyjj13uas1rant8tm4e4"      # from Pushover "Your Applications"
USER_KEY = "u9w3eapuf49w2oijy3y7hiimab3d1f"        # from top of Pushover dashboard

def send_pushover_message(title: str, message: str):
    data = {
        "token": TOKEN,
        "user": USER_KEY,
        "title": title,
        "message": message,
    }

    response = requests.post(PUSHOVER_API_URL, data=data)

    # Simple error handling
    try:
        response.raise_for_status()
        print("Notification sent! Response:", response.json())
    except requests.exceptions.HTTPError as e:
        print("Error sending notification:", e)
        print("Response content:", response.text)


if __name__ == "__main__":
    send_pushover_message(
        title="Hello from Python 👋",
        message="This is a test notification using Pushover!"
    )