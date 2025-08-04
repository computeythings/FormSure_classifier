#!/usr/bin/env python3
"""
Local development server for Document Region Classification Service
Simulates AWS Lambda environment for local testing
"""

import os
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import signal
import sys

# Set up environment for local development
os.environ.setdefault('AWS_REGION', 'us-east-1')
os.environ.setdefault('TABLE_PREFIX', 'classifier')
os.environ.setdefault('MODEL_S3_BUCKET', 'local-test-bucket')
os.environ.setdefault('DEBUG_MODE', 'true')
os.environ.setdefault('LOCAL_MODE', 'true')

# Import the classifier after setting environment
from classifier import lambda_handler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LocalLambdaHandler(BaseHTTPRequestHandler):
    """HTTP handler that simulates AWS Lambda + API Gateway"""
    
    def _set_cors_headers(self):
        """Set CORS headers for local development"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    
    def _send_response(self, status_code, body, content_type='application/json'):
        """Send HTTP response"""
        self.send_response(status_code)
        self.send_header('Content-Type', content_type)
        self._set_cors_headers()
        self.end_headers()
        
        if isinstance(body, dict):
            body = json.dumps(body)
        
        self.wfile.write(body.encode('utf-8'))
    
    def _create_lambda_event(self, method, path, body=None, query_params=None):
        """Create Lambda event object from HTTP request"""
        return {
            'httpMethod': method,
            'path': path,
            'headers': {
                'Content-Type': self.headers.get('Content-Type', ''),
                'User-Agent': self.headers.get('User-Agent', ''),
            },
            'queryStringParameters': query_params or {},
            'body': body,
            'isBase64Encoded': False,
            'requestContext': {
                'requestId': 'local-request',
                'stage': 'local',
                'httpMethod': method,
                'path': path
            }
        }
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self._send_response(200, {'message': 'CORS preflight'})
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query_params = parse_qs(parsed_url.query)
        
        # Flatten query parameters
        flat_params = {k: v[0] if v else '' for k, v in query_params.items()}
        
        logger.info(f"GET {path}")
        
        try:
            # Create Lambda event
            event = self._create_lambda_event('GET', path, query_params=flat_params)
            
            # Call Lambda handler
            response = lambda_handler(event, None)
            
            # Send response
            self._send_response(
                response['statusCode'],
                response['body'],
                response['headers'].get('Content-Type', 'application/json')
            )
            
        except Exception as e:
            logger.error(f"Error handling GET {path}: {e}")
            self._send_response(500, {'error': str(e)})
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else None
        
        logger.info(f"POST {path}")
        
        try:
            # Create Lambda event
            event = self._create_lambda_event('POST', path, body=body)
            
            # Call Lambda handler
            response = lambda_handler(event, None)
            
            # Send response
            self._send_response(
                response['statusCode'],
                response['body'],
                response['headers'].get('Content-Type', 'application/json')
            )
            
        except Exception as e:
            logger.error(f"Error handling POST {path}: {e}")
            self._send_response(500, {'error': str(e)})
    
    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info(f"{self.address_string()} - {format % args}")

class LocalServer:
    """Local development server"""
    
    def __init__(self, host='localhost', port=8000):
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        
    def start(self):
        """Start the server"""
        try:
            self.server = HTTPServer((self.host, self.port), LocalLambdaHandler)
            logger.info(f"Starting local server on http://{self.host}:{self.port}")
            
            # Print available endpoints
            logger.info("Available endpoints:")
            logger.info(f"  GET  http://{self.host}:{self.port}/health")
            logger.info(f"  GET  http://{self.host}:{self.port}/stats") 
            logger.info(f"  POST http://{self.host}:{self.port}/classify")
            logger.info(f"  POST http://{self.host}:{self.port}/feedback")
            logger.info(f"  POST http://{self.host}:{self.port}/retrain")
            
            # Start server in thread
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            logger.info("Server started successfully!")
            logger.info("Press Ctrl+C to stop the server")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    def stop(self):
        """Stop the server"""
        if self.server:
            logger.info("Shutting down server...")
            self.server.shutdown()
            self.server.server_close()
            if self.server_thread:
                self.server_thread.join(timeout=5)
            logger.info("Server stopped")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal")
    sys.exit(0)

def main():
    """Main entry point"""
    # Handle command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Local Document Classifier Server')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        os.environ['DEBUG_MODE'] = 'true'
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start server
    server = LocalServer(args.host, args.port)
    
    if server.start():
        try:
            # Keep main thread alive
            while True:
                signal.pause()
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()