import json
from http import server


class CustomHandler(server.SimpleHTTPRequestHandler):

    def do_GET(self):
        self.send_response(400)
        self.send_header('content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"CSV UPLOADING")

    def do_POST(self):
        self.send_response(200)
        post_data = (self.rfile.read(int(self.headers['content-length'])))
        global NumberWav
        global NumberPdm
        global Path
        NumberWav = NumberWav + 1
        NumberPdm = NumberPdm + 1
        if self.path.endswith("/uploadAudio"):
            f = open(Path + "\Данные с микрофона " + str(NumberWav) + ".wav", 'wb')
            f.write(post_data)
            f.close()

        if self.path.endswith("/uploadPCM"):
            content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
            print(content_length);
            fa = open(Path + "\Данные с микрофона " + str(NumberPdm) + ".doc", 'wb')
            fa.write(post_data)
            fa.close()

        self.send_header('content-type', 'application/json')
        self.send_header('Server', 'CoolServer')
        self.end_headers()
        self.wfile.write(json.dumps({'result': True}).encode())
        print(" ")
        if NumberWav > 1001:
            NumberWav = 0
        if NumberPdm > 1001:
            NumberPdm = 0

    def do_PUT(self):
        self.send_response(200)
        self.send_header('content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'PUT request\n')

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()