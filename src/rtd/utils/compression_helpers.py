import cv2
import struct
import numpy as np

def send_compressed(sock, img, quality=90):
    """
    Compress an image using JPEG and send it over the socket.
    """
    ret, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ret:
        raise ValueError("Image compression failed")
    data = buf.tobytes()
    # Prefix the data with its length
    msg = struct.pack('!I', len(data)) + data
    sock.sendall(msg)

def recvall(sock, n):
    """
    Helper function to receive exactly n bytes from the socket.
    """
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def recv_compressed(sock):
    """
    Receive a compressed image from the socket and decompress it using OpenCV.
    """
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('!I', raw_msglen)[0]
    data = recvall(sock, msglen)
    if data is None:
        return None
    # Convert the byte data to a NumPy array and decode the image
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img 