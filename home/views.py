import json
import time
from django.shortcuts import render
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

# Create your views here.

# def home(request):
#     for i in range(1,10):
#         channel_layer = get_channel_layer()
#         data = {"count":i}
#         async_to_sync(channel_layer.group_send)(
#             "test_consumer_group",{
#                 'type' : 'send_notification',
#                 'value' : json.dumps(data)
#             }
#         )
#         time.sleep(1)
#     return render(request,"home.html")

async def home(request):
    for i in range(1,10):
        channel_layer = get_channel_layer()
        data = {"count":i}
        await (channel_layer.group_send)(
            "new_consumer_group",{
                'type' : 'send_notification',
                'value' : json.dumps(data)
            }
        )
        time.sleep(1)
    return render(request,"home_video_stream.html")