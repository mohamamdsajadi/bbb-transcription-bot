import os
from queue import Queue
import threading
import uuid

# Client class for each connected Client to handle the data seperatly.
class Client:
    def __init__(self, client):

        self.id = str(uuid.uuid4())

        self.mutex = threading.Lock()

        self._client = client
        self._instance = None

    def send(self, data):
        self._client.send_message(data)

    def stop(self):
        with self.mutex:
            self._client.stop()
            self.data_queue = Queue()
            self.last_sample = bytes()