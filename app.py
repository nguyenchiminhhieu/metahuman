# server.py
from flask import Flask, render_template, send_from_directory, request, jsonify
from flask_sockets import Sockets
import base64
import time
import json
import gevent
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
import os
import re
import numpy as np
from threading import Thread, Event
import multiprocessing

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription
from webrtc import HumanPlayer

import argparse

import shutil
import asyncio
import websockets

app = Flask(__name__)
sockets = Sockets(app)
global nerfreal

@sockets.route('/humanecho')
def echo_socket(ws):
    if not ws:
        print('未建立连接！')
        return 'Please use WebSocket'
    else:
        print('建立连接！')
        while True:
            message = ws.receive()
            if not message or len(message) == 0:
                return '输入信息为空'
            else:
                nerfreal.put_msg_txt(message)

def llm_response(message):
    from llm.LLM import LLM
    llm = LLM().init_model('VllmGPT', model_path='THUDM/chatglm3-6b')
    response = llm.chat(message)
    print(response)
    return response

@sockets.route('/humanchat')
def chat_socket(ws):
    if not ws:
        print('未建立连接！')
        return 'Please use WebSocket'
    else:
        print('建立连接！')
        while True:
            message = ws.receive()
            if len(message) == 0:
                return '输入信息为空'
            else:
                res = llm_response(message)
                nerfreal.put_msg_txt(res)

#####webrtc###############################
pcs = set()

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    player = HumanPlayer(nerfreal)
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )

async def human(request):
    params = await request.json()

    if params['type'] == 'echo':
        nerfreal.put_msg_txt(params['text'])
    elif params['type'] == 'chat':
        res = await asyncio.get_event_loop().run_in_executor(None, llm_response, params['text'])
        nerfreal.put_msg_txt(res)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data": "ok"}
        ),
    )

async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

async def post(url, data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        print(f'Error: {e}')

async def run(push_url):
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    player = HumanPlayer(nerfreal)
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    await pc.setLocalDescription(await pc.createOffer())
    answer = await post(push_url, pc.localDescription.sdp)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type='answer'))

##########################################
# WebSocket client code
async def websocket_client():
    uri = "ws://localhost:10002"  # WebSocket服务端地址
    async with websockets.connect(uri) as websocket:
        await websocket.send("Hello, server!")
        while True:
            message = await websocket.recv()
            print(f"Received message from server: {message}")

def start_websocket_client():
    asyncio.new_event_loop().run_until_complete(websocket_client())

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose', type=str, default="data/data_kf.json", help="transforms.json, pose source")
    parser.add_argument('--au', type=str, default="data/au.csv", help="eye blink area")
    parser.add_argument('--torso_imgs', type=str, default="", help="torso images path")
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --exp_eye")
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1], help="data range to use")
    parser.add_argument('--workspace', type=str, default='data/video')
    parser.add_argument('--seed', type=int, default=0)

    # ... 省略其余解析参数的代码 ...

    opt = parser.parse_args()

    if opt.model == 'ernerf':
        from ernerf.nerf_triplane.provider import NeRFDataset_Test
        from ernerf.nerf_triplane.utils import *
        from ernerf.nerf_triplane.network import NeRFNetwork
        from nerfreal import NeRFReal

        opt.test = True
        opt.test_train = False
        opt.fp16 = True
        opt.cuda_ray = True
        opt.exp_eye = True
        opt.smooth_eye = True

        if opt.torso_imgs == '':
            opt.torso = True

        opt.asr = True

        if opt.patch_size > 1:
            assert opt.num_rays % (opt.patch_size ** 2) == 0
        seed_everything(opt.seed)
        print(opt)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = NeRFNetwork(opt)

        criterion = torch.nn.MSELoss(reduction='none')
        metrics = []
        print(model)
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)

        test_loader = NeRFDataset_Test(opt, device=device).dataloader()
        model.aud_features = test_loader._data.auds
        model.eye_areas = test_loader._data.eye_area

        nerfreal = NeRFReal(opt, trainer, test_loader)
    elif opt.model == 'musetalk':
        from musereal import MuseReal
        print(opt)
        nerfreal = MuseReal(opt)
    elif opt.model == 'wav2lip':
        from lipreal import LipReal
        print(opt)
        nerfreal = LipReal(opt)

    if opt.transport == 'rtmp':
        thread_quit = Event()
        rendthrd = Thread(target=nerfreal.render, args=(thread_quit,))
        rendthrd.start()

    #############################################################################
    appasync = web.Application()
    appasync.on_shutdown.append(on_shutdown)
    appasync.router.add_post("/offer", offer)
    appasync.router.add_post("/human", human)
    appasync.router.add_static('/', path='web')

    cors = aiohttp_cors.setup(appasync, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })
    for route in list(appasync.router.routes()):
        cors.add(route)

    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        loop.run_until_complete(site.start())
        if opt.transport == 'rtcpush':
            loop.run_until_complete(run(opt.push_url))
        loop.run_forever()

    Thread(target=run_server, args=(web.AppRunner(appasync),)).start()

    # Start WebSocket client in a new thread
    Thread(target=start_websocket_client).start()

    print('start websocket server')
    #app.on_shutdown.append(on_shutdown)
    #app.router.add_post("/offer", offer)
    server = pywsgi.WSGIServer(('0.0.0.0', 8000), app, handler_class=WebSocketHandler)
    server.serve_forever()
    
    