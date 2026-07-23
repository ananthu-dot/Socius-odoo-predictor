import http.server
import socketserver
import json
import os
import urllib.parse
import pandas as pd

from api.revenue import api_train_revenue, api_predict_revenue, api_forecast_revenue
from api.products import api_train_product, api_predict_product, api_forecast_product, api_forecast_category
from api.customers import api_train_customer_model, api_analyze_customers
from generate_sample_data import generate_demo_dataset

PORT = 8000
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

class PredictiveAppHandler(http.server.SimpleHTTPRequestHandler):

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0]}")

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/":
            return self.serve_file(os.path.join(STATIC_DIR, "index.html"), "text/html")
        elif path.startswith("/static/"):
            rel_path = path[len("/static/"):]
            file_path = os.path.join(STATIC_DIR, rel_path)
            content_type = "text/css" if path.endswith(".css") else ("application/javascript" if path.endswith(".js") else "text/html")
            return self.serve_file(file_path, content_type)
        elif path == "/api/sample-data":
            return self.handle_sample_data()
        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length)
        
        try:
            payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception as e:
            return self.send_json_response({"error": f"Invalid JSON body: {str(e)}"}, status=400)

        # Route matching
        if path == "/api/revenue/predict":
            res = api_predict_revenue(payload)
        elif path == "/api/revenue/forecast":
            res = api_forecast_revenue(payload)
        elif path == "/api/revenue/train":
            res = api_train_revenue(payload)
        elif path == "/api/products/predict":
            res = api_predict_product(payload)
        elif path == "/api/products/forecast":
            res = api_forecast_product(payload)
        elif path == "/api/products/forecast/category":
            res = api_forecast_category(payload)
        elif path == "/api/products/train":
            res = api_train_product(payload)
        elif path == "/api/customers/analyze":
            res = api_analyze_customers(payload)
        elif path == "/api/customers/train":
            res = api_train_customer_model(payload)
        else:
            return self.send_json_response({"error": "Unknown API endpoint"}, status=404)

        status_code = 400 if "error" in res else 200
        return self.send_json_response(res, status=status_code)

    def handle_sample_data(self):
        try:
            orders_df, order_lines_df = generate_demo_dataset(months=24)
            res = {
                "status": "success",
                "orders": orders_df.to_dict(orient="records"),
                "order_lines": order_lines_df.to_dict(orient="records"),
                "summary": {
                    "total_orders": len(orders_df),
                    "total_order_lines": len(order_lines_df),
                    "start_date": str(orders_df["order_date"].min()),
                    "end_date": str(orders_df["order_date"].max()),
                    "unique_customers": int(orders_df["customer"].nunique()),
                    "unique_products": int(order_lines_df["product_name"].nunique())
                }
            }
            return self.send_json_response(res)
        except Exception as e:
            return self.send_json_response({"error": f"Failed to generate sample data: {str(e)}"}, status=500)

    def serve_file(self, file_path, content_type):
        if not os.path.exists(file_path):
            self.send_error(404, f"File not found: {file_path}")
            return
        
        with open(file_path, "rb") as f:
            content = f.read()

        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json_response(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), PredictiveAppHandler) as httpd:
        print(f"Socius Predictor Dashboard Server running at http://127.0.0.1:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
            httpd.server_close()

if __name__ == "__main__":
    run_server()
