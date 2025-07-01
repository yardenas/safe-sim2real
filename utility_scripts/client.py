import cloudpickle as pickle
import zmq


def send_request(address="tcp://localhost:5555"):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(address)
    # Serialize policy and number of steps
    message = pickle.dumps(("Hi", 1000))
    print("Sending request to server...")
    socket.send(message)
    # Receive and deserialize response
    response = socket.recv()
    print("Received response from server...")
    transitions = pickle.loads(response)
    print(f"Received {len(transitions)} transitions...")

    socket.close()
    context.term()


if __name__ == "__main__":
    send_request()
