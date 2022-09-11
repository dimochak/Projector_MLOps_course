from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
from sklearn.linear_model import LinearRegression


class HttpGetHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3
        reg = LinearRegression().fit(X, y)
        self.wfile.write('<html><head><meta charset="utf-8">'.encode())
        self.wfile.write('<title>Linear regression</title></head>'.encode())
        self.wfile.write(f'<body>'
                         f'<p>REGRESSION SCORE: {reg.score(X, y)}</p>'
                         f'<p>REGRESSION COEFFICIENTS: {reg.coef_}</p>'
                         f'<p>REGRESSION INTERCEPT: {reg.intercept_}</p>'
                         f'<p>REGRESSION PREDICTION @(3, 5): {reg.predict(np.array([[3, 5]]))}</p></body></html>'.encode())


def run(server_class=HTTPServer, handler_class=HttpGetHandler):
        server_address = ("0.0.0.0", 8000)
        httpd = server_class(server_address, handler_class)
        try:
            print('LAUNCHING SERVER')
            httpd.serve_forever()
        except KeyboardInterrupt:
            httpd.server_close()


if __name__ == '__main__':
    run()
