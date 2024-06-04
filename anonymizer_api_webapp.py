from asgiref.wsgi import WsgiToAsgi
from fastapi import FastAPI
from starlette.responses import RedirectResponse

# Import the Flask app
from anonymizer_flask_app import app as wsgi_flask_app


# Disable flask logging
from flask.logging import default_handler
wsgi_flask_app.logger.removeHandler(default_handler)

# Import the FastAPI app
from anonymizer_api_app import anonymizer_api as fastapi_app

# Create a main FastAPI app
main_app = FastAPI()

asgi_flask_app = WsgiToAsgi(wsgi_flask_app)

main_app.mount("/web", asgi_flask_app)
main_app.mount("/api", fastapi_app)

@main_app.get("/")
def index():
    # redirect to /web
    return RedirectResponse("/web")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(main_app, host="0.0.0.0", port=8000)