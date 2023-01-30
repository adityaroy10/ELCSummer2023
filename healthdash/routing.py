from channels.routing import ProtocolTypeRouter,URLRouter
from channels.auth import AuthMiddlewareStack
from django.urls import path
from dashboard import consumer

websocket_urlPattern=[
    path(r'ws/polData/',consumer.DashConsumer),
    path(r'ws/predData/',consumer.PredConsumer),
]

application = ProtocolTypeRouter({
    'websocket':AuthMiddlewareStack(URLRouter(websocket_urlPattern))
})
