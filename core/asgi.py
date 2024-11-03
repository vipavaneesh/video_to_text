"""
ASGI config for core project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application
from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.urls import path
from home.consumers import *

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")

application = get_asgi_application()

ws_patterns = [

    path('ws/test/',TestConsumer.as_asgi()),
    path('ws/new/',NewConsumer.as_asgi()),
    path('ws/video/',VideoConsumer.as_asgi())

]

# Create the ASGI application with routing # The below one is correct one 1333 29102024
application = ProtocolTypeRouter({
    "http": application,  # Handle HTTP requests
    "websocket": AuthMiddlewareStack(  # Handle WebSocket requests with authentication
        URLRouter(ws_patterns)
    ),
})

# application = ProtocolTypeRouter({
    
#     'websocket': URLRouter(ws_patterns)
# })

# application = ProtocolTypeRouter({
#     "http": get_asgi_application(),  # Handle HTTP requests
#     "websocket": AuthMiddlewareStack(  # Handle WebSocket requests
#         URLRouter([
#             path('ws/test/', TestConsumer.as_asgi()),  # Ensure this path matches your WebSocket URL
#         ])
#     ),
# })