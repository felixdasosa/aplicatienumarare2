from flask import Flask, render_template_string, jsonify, request, Response
from ultralytics import YOLO
import cv2
import threading
import json
import time
import os
import numpy as np
from datetime import datetime

# --- SETĂRI GLOBALE ---
SKIP_FRAMES = 2
CONF_THRESH = 0.30 
CONFIG_FILE = "cameras.json"

app = Flask(__name__)

# --- STRUCTURI DE DATE DINAMICE ---
cameras_config = {}  
camera_data = {}     
history_data = {}    
thread_flags = {}    
latest_frames = {}   

TIME_SLOTS = [f"{h:02d}:00-{h+2:02d}:00" for h in range(0, 24, 2)]

def get_current_slot():
    now = datetime.now()
    start_h = (now.hour // 2) * 2
    return f"{start_h:02d}:00-{start_h+2:02d}:00"

# --- MANAGEMENT CONFIGURAȚIE ---
def load_config():
    global cameras_config
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                cameras_config = json.load(f)
        except:
            cameras_config = {}
    else:
        cameras_config = {}

def save_config():
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cameras_config, f, indent=4)

def init_camera_structures(cam_id):
    if cam_id not in camera_data:
        camera_data[cam_id] = {"in": 0, "out": 0, "status": "Init...", "name": cameras_config[cam_id].get("name", cam_id)}
    if cam_id not in history_data:
        history_data[cam_id] = {slot: {"in": 0, "out": 0} for slot in TIME_SLOTS}

# --- MATEMATICĂ ---
def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

# --- PROCESARE VIDEO ---
def process_camera(cam_id):
    model = YOLO('yolo11n.pt') 
    print(f"[THREAD] Start Camera {cam_id}")
    cap = None
    
    while thread_flags.get(cam_id, False):
        config = cameras_config.get(cam_id)
        if not config: break
        
        # LOGICĂ PENTRU LINK RTSP (DVR vs CAMERĂ IP)
        canal_dvr = config.get('channel', '').strip()
        if canal_dvr:
            path = f"/Streaming/Channels/{canal_dvr}01"
        else:
            path = config.get('rtsp_path', '/Streaming/Channels/101').strip()
            if not path:
                path = '/Streaming/Channels/101' # Default fallback
                
        if not path.startswith('/'): 
            path = '/' + path
            
        url = f"rtsp://{config['user']}:{config['password']}@{config['ip']}:{config['port']}{path}"
        
        scale_x = 640 / 1280.0
        scale_y = 360 / 720.0

        track_history = {} 
        counted_ids = []
        frame_count = 0
        
        cap = cv2.VideoCapture(url)
        
        while thread_flags.get(cam_id, False):
            ret, frame = cap.read()
            if not ret:
                camera_data[cam_id]["status"] = "OFFLINE"
                cap.release()
                time.sleep(5)
                if thread_flags.get(cam_id, False):
                    cap = cv2.VideoCapture(url)
                continue
                
            camera_data[cam_id]["status"] = "ONLINE"
            camera_data[cam_id]["name"] = config["name"]
            
            # Actualizare Live a Liniei
            current_config = cameras_config.get(cam_id, config)
            line_start = tuple(current_config.get('pt1', [0,0]))
            line_end = tuple(current_config.get('pt2', [0,0]))
            L1 = (int(line_start[0] * scale_x), int(line_start[1] * scale_y))
            L2 = (int(line_end[0] * scale_x), int(line_end[1] * scale_y))

            frame_small = cv2.resize(frame, (640, 360))
            
            frame_disp = frame_small.copy()
            cv2.line(frame_disp, L1, L2, (0, 255, 0), 2)
            ret_jpg, buffer = cv2.imencode('.jpg', frame_disp)
            if ret_jpg: latest_frames[cam_id] = buffer.tobytes()
            
            frame_count += 1
            if frame_count % SKIP_FRAMES != 0: continue 
            
            results = model.track(frame_small, persist=True, verbose=False, imgsz=320, conf=CONF_THRESH, classes=[0], tracker="bytetrack.yaml")
            current_ids_in_frame = []

            if results and results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, ids):
                    current_ids_in_frame.append(track_id)
                    x1, y1, x2, y2 = map(int, box)
                    curr_point = ((x1 + x2) // 2, y2)

                    if track_id in track_history and track_id not in counted_ids:
                        prev_point = track_history[track_id]
                        if intersect(prev_point, curr_point, L1, L2):
                            current_slot = get_current_slot()
                            line_y_at_x = L1[1] + (L2[1] - L1[1]) * (curr_point[0] - L1[0]) / (L2[0] - L1[0] + 0.001)
                            
                            if prev_point[1] < line_y_at_x and curr_point[1] > line_y_at_x:
                                camera_data[cam_id]["in"] += 1
                                history_data[cam_id][current_slot]["in"] += 1
                                counted_ids.append(track_id)
                                
                            elif prev_point[1] > line_y_at_x and curr_point[1] < line_y_at_x:
                                camera_data[cam_id]["out"] += 1
                                history_data[cam_id][current_slot]["out"] += 1
                                counted_ids.append(track_id)

                    track_history[track_id] = curr_point
                    
            if frame_count % 100 == 0:
                track_history = {k: v for k, v in track_history.items() if k in current_ids_in_frame}
                
            time.sleep(0.01)
            
        if cap: cap.release()
    print(f"[THREAD] Stop Camera {cam_id}")

def start_camera_thread(cam_id):
    if cam_id not in thread_flags or not thread_flags[cam_id]:
        thread_flags[cam_id] = True
        t = threading.Thread(target=process_camera, args=(cam_id,))
        t.daemon = True
        t.start()

def stop_camera_thread(cam_id):
    if cam_id in thread_flags: thread_flags[cam_id] = False

# --- API ENDPOINTS ---
@app.route('/data')
def data(): return jsonify({"live": camera_data, "history": history_data})

@app.route('/get_cameras')
def get_cameras(): return jsonify(cameras_config)

@app.route('/save_camera', methods=['POST'])
def save_camera():
    data = request.json
    cam_id = data.get('id')
    is_edit = cam_id in cameras_config
    
    if is_edit:
        stop_camera_thread(cam_id)
        time.sleep(1) 
        
    old_pt1 = cameras_config.get(cam_id, {}).get("pt1", [0, 0])
    old_pt2 = cameras_config.get(cam_id, {}).get("pt2", [0, 0])

    cameras_config[cam_id] = {
        "name": data.get('name'),
        "ip": data.get('ip'),
        "port": data.get('port'),
        "user": data.get('user'),
        "password": data.get('password'),
        "channel": data.get('channel', ''),
        "rtsp_path": data.get('rtsp_path', ''),
        "pt1": old_pt1,
        "pt2": old_pt2
    }
    
    save_config()
    init_camera_structures(cam_id)
    start_camera_thread(cam_id)
    return jsonify({"success": True, "id": cam_id})

@app.route('/save_line', methods=['POST'])
def save_line():
    data = request.json
    cam_id = data.get('id')
    if cam_id in cameras_config:
        cameras_config[cam_id]['pt1'] = data.get('pt1')
        cameras_config[cam_id]['pt2'] = data.get('pt2')
        save_config()
        return jsonify({"success": True})
    return jsonify({"success": False})

@app.route('/delete_camera', methods=['POST'])
def delete_camera():
    cam_id = request.json.get('id')
    if cam_id in cameras_config:
        stop_camera_thread(cam_id)
        del cameras_config[cam_id]
        if cam_id in camera_data: del camera_data[cam_id]
        if cam_id in history_data: del history_data[cam_id]
        save_config()
        return jsonify({"success": True})
    return jsonify({"success": False})

def generate_frames(cam_id):
    while True:
        frame = latest_frames.get(cam_id)
        if frame:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            img = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(img, "Se conecteaza la camera...", (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            _, buffer = cv2.imencode('.jpg', img)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.1)

@app.route('/video_feed/<cam_id>')
def video_feed(cam_id): return Response(generate_frames(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- WEB INTERFACE ---
HTML_PAGE = """
<!DOCTYPE html>
<html lang="ro">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Monitorizare & Configurare Camere</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background-color: #0f0f0f; color: #fff; margin: 0; padding: 20px; }
        h1 { text-align: center; color: #4facfe; margin-bottom: 20px; }
        .tabs { display: flex; justify-content: center; margin-bottom: 20px; flex-wrap: wrap; gap: 10px; }
        .tab-btn { background: #333; color: white; border: none; padding: 10px 20px; cursor: pointer; font-size: 1.1em; border-radius: 5px; transition: 0.3s; }
        .tab-btn.active { background: #4facfe; color: black; font-weight: bold; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        /* LIVE GRID */
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; max-width: 1200px; margin: 0 auto; }
        .card { background: #1e1e1e; border-radius: 15px; padding: 20px; box-shadow: 0 10px 20px rgba(0,0,0,0.5); border-top: 5px solid #333; }
        .online { border-top-color: #00e676; }
        .offline { border-top-color: #ff1744; }
        .stats { display: flex; justify-content: space-around; margin-bottom: 15px; }
        .val { font-size: 2.5em; font-weight: bold; text-align: center;}
        .lbl { font-size: 0.8em; color: #aaa; text-align: center; }
        .in { color: #00e676; }
        .out { color: #ff1744; }
        .total-box { background: #333; border-radius: 10px; padding: 10px; text-align: center; margin-top: 10px; margin-bottom: 15px; }
        .val-total { font-size: 2em; font-weight: bold; color: #4facfe; }

        .btn-play { background: #2196F3; color: white; border: none; padding: 10px; width: 100%; border-radius: 5px; cursor: pointer; font-weight: bold; font-size: 1em; transition: 0.2s; }
        .btn-play:hover { background: #1976D2; }

        /* TABLES & FORMS */
        table { width: 100%; max-width: 1200px; margin: 0 auto; border-collapse: collapse; background: #1e1e1e; }
        th, td { padding: 12px; text-align: center; border-bottom: 1px solid #444; }
        th { background: #333; color: #4facfe; }
        tr:hover { background: #2a2a2a; }
        .h-in { color: #00e676; }
        .h-out { color: #ff1744; }
        
        .form-container { background: #1e1e1e; padding: 20px; border-radius: 10px; max-width: 800px; margin: 0 auto 30px auto; }
        .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        input { width: 100%; padding: 10px; background: #333; border: 1px solid #555; color: white; border-radius: 5px; box-sizing: border-box; }
        .btn-submit { background: #00e676; color: black; font-weight: bold; border: none; padding: 12px; border-radius: 5px; cursor: pointer; width: 100%; margin-top: 15px; }
        
        .btn-edit { background: #4facfe; color: black; font-weight:bold; border: none; padding: 6px 12px; border-radius: 3px; cursor: pointer; margin:2px; }
        .btn-draw { background: #ffb300; color: black; font-weight:bold; border: none; padding: 6px 12px; border-radius: 3px; cursor: pointer; margin:2px; }
        .btn-delete { background: #ff1744; color: white; font-weight:bold; border: none; padding: 6px 12px; border-radius: 3px; cursor: pointer; margin:2px; }
        
        label { font-size: 0.9em; color: #aaa; margin-bottom: 5px; display: block; }
        .info-text { font-size: 0.8em; color: #888; margin-top: 3px; }
        .optional-box { border: 1px solid #555; padding: 10px; border-radius: 5px; background: #252525;}

        /* MODALE */
        .modal-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); z-index: 1000; text-align: center; }
        .modal-content { position: relative; display: inline-block; margin-top: 50px; background: #222; padding: 20px; border-radius: 10px; }
        canvas { position: absolute; top: 20px; left: 20px; cursor: crosshair; }
        .modal-buttons { margin-top: 20px; }
        .btn-save-line { background: #00e676; color: black; padding: 10px 20px; border: none; border-radius: 5px; font-weight: bold; font-size: 1.1em; cursor: pointer; }
        .btn-cancel { background: #ff1744; color: white; padding: 10px 20px; border: none; border-radius: 5px; font-weight: bold; font-size: 1.1em; cursor: pointer; margin-left: 10px; }
    </style>
</head>
<body>
    <h1>Panou Monitorizare Trafic</h1>

    <div class="tabs">
        <button id="btn-live" class="tab-btn active" onclick="switchTab('live')">Vizualizare Live</button>
        <button id="btn-history" class="tab-btn" onclick="switchTab('history')">Istoric (Intervale 2h)</button>
        <button id="btn-config" class="tab-btn" onclick="switchTab('config'); loadSettings();">Setări Camere</button>
    </div>

    <div id="live" class="tab-content active"><div class="grid" id="live-grid"></div></div>

    <div id="history" class="tab-content">
        <table id="history-table"><thead id="history-head"></thead><tbody id="history-body"></tbody></table>
    </div>

    <div id="config" class="tab-content">
        <div class="form-container">
            <h3 id="form-title" style="margin-top:0;">Adaugă / Editează Cameră</h3>
            <div class="form-grid">
                <div><label>ID Unic (ex: cam1)</label><input type="text" id="f_id" placeholder="cam_intrare"></div>
                <div><label>Nume Afișat</label><input type="text" id="f_name" placeholder="Intrare Principala"></div>
                <div><label>IP Cameră / DVR</label><input type="text" id="f_ip" placeholder="82.76.164.107"></div>
                <div><label>Port RTSP</label><input type="text" id="f_port" placeholder="10554"></div>
                <div><label>Utilizator</label><input type="text" id="f_user" placeholder="admin"></div>
                <div><label>Parolă</label><input type="password" id="f_pass" placeholder="parola123"></div>
            </div>
            
            <h4 style="margin-bottom: 5px; color: #4facfe;">Setări Video (Completati doar una din cele de mai jos)</h4>
            <div class="form-grid optional-box">
                <div>
                    <label>Dacă folosiți DVR: Număr Canal</label>
                    <input type="text" id="f_channel" placeholder="6">
                    <div class="info-text">Ex: Puneți 6 pentru camera 6 de pe DVR. (Lasati gol pt Camera IP directă)</div>
                </div>
                <div>
                    <label>Dacă folosiți Cameră IP: Cale RTSP</label>
                    <input type="text" id="f_rtsp_path" placeholder="/Streaming/Channels/101">
                    <div class="info-text">Ex: /Streaming/Channels/101 pt Hikvision IP. (Lasati gol daca e conectata la DVR)</div>
                </div>
            </div>

            <button class="btn-submit" onclick="saveCamera()">Salvează Camera</button>
        </div>

        <table>
            <thead><tr><th>ID</th><th>Nume</th><th>IP:Port</th><th>Conexiune</th><th>Acțiuni</th></tr></thead>
            <tbody id="config-table-body"></tbody>
        </table>
    </div>

    <div id="view-modal" class="modal-overlay">
        <div class="modal-content">
            <h2 id="view-modal-title" style="margin-top:0; color:#4facfe;">Vizualizare Live</h2>
            <img id="view-stream-img" width="640" height="360" style="display:block; border-radius:5px; margin: 0 auto;" />
            <div class="modal-buttons"><button class="btn-cancel" onclick="closeViewModal()">Închide</button></div>
        </div>
    </div>

    <div id="draw-modal" class="modal-overlay">
        <div class="modal-content">
            <h2 style="margin-top:0; color:#ffb300;">Editează Linia de Detecție</h2>
            <p style="color:#aaa; font-size:0.9em;">Trage cu mouse-ul peste imagine pentru a plasa linia de detecție.</p>
            <div style="position:relative; width:640px; height:360px; margin: 0 auto;">
                <img id="stream-img" width="640" height="360" style="display:block; border-radius:5px;" />
                <canvas id="draw-canvas" width="640" height="360"></canvas>
            </div>
            <div class="modal-buttons">
                <button class="btn-save-line" onclick="saveDrawnLine()">Salvează Linia</button>
                <button class="btn-cancel" onclick="closeDrawModal()">Anulează</button>
            </div>
        </div>
    </div>

    <script>
        let currentCameras = {};
        let currentLiveDataCache = {};
        let activeDrawCamId = null;

        // Desenare Canvas
        let canvas = document.getElementById('draw-canvas');
        let ctx = canvas.getContext('2d');
        let isDrawing = false;
        let startX = 0, startY = 0, endX = 0, endY = 0;

        canvas.addEventListener('mousedown', (e) => {
            const rect = canvas.getBoundingClientRect();
            startX = e.clientX - rect.left; startY = e.clientY - rect.top;
            isDrawing = true;
        });

        canvas.addEventListener('mousemove', (e) => {
            if (!isDrawing) return;
            const rect = canvas.getBoundingClientRect();
            let currentX = e.clientX - rect.left; let currentY = e.clientY - rect.top;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.beginPath(); ctx.moveTo(startX, startY); ctx.lineTo(currentX, currentY);
            ctx.strokeStyle = '#00e676'; ctx.lineWidth = 3; ctx.stroke();
        });

        canvas.addEventListener('mouseup', (e) => {
            if (!isDrawing) return;
            const rect = canvas.getBoundingClientRect();
            endX = e.clientX - rect.left; endY = e.clientY - rect.top;
            isDrawing = false;
        });

        function switchTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabName).classList.add('active');
            document.getElementById('btn-' + tabName).classList.add('active');
        }

        function updateData() {
            fetch('/data').then(r => r.json()).then(data => {
                const live = data.live;
                const history = data.history;
                currentLiveDataCache = live; 
                const camIds = Object.keys(live);

                // RENDER LIVE GRID
                const grid = document.getElementById('live-grid');
                camIds.forEach(id => {
                    const info = live[id];
                    let card = document.getElementById(`card-${id}`);
                    if (!card) {
                        card = document.createElement('div');
                        card.id = `card-${id}`;
                        grid.appendChild(card);
                    }
                    const statusClass = info.status === "ONLINE" ? "online" : "offline";
                    const statusDot = info.status === "ONLINE" ? "🟢 ONLINE" : "🔴 OFFLINE";

                    card.className = `card ${statusClass}`;
                    card.innerHTML = `
                        <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                            <span style="font-weight:bold;">${info.name}</span>
                            <span style="font-size:0.8em; color:#aaa;">${statusDot}</span>
                        </div>
                        <div class="stats">
                            <div><div class="val in">${info.in}</div><div class="lbl">INTRĂRI</div></div>
                            <div><div class="val out">${info.out}</div><div class="lbl">IEȘIRI</div></div>
                        </div>
                        <div class="total-box"><div class="val-total">${info.in - info.out}</div><div style="font-size:0.8em;">TOTAL PERSOANE</div></div>
                        <button class="btn-play" onclick="openViewModal('${id}')">▶ Vezi Live</button>
                    `;
                });
                Array.from(grid.children).forEach(child => {
                    const cid = child.id.replace('card-', '');
                    if(!camIds.includes(cid)) grid.removeChild(child);
                });

                // RENDER HISTORY TABLE
                if(camIds.length > 0) {
                    const thead = document.getElementById('history-head');
                    const tbody = document.getElementById('history-body');
                    let headHtml1 = `<tr><th rowspan="2">Interval Orar</th>`;
                    let headHtml2 = `<tr>`;
                    camIds.forEach(id => {
                        headHtml1 += `<th colspan="3">${live[id].name}</th><th style="width:5px;"></th>`;
                        headHtml2 += `<th style="font-size:0.8em;">IN</th><th style="font-size:0.8em;">OUT</th><th style="font-size:0.8em;">TOT</th><th></th>`;
                    });
                    headHtml1 += `</tr>`; headHtml2 += `</tr>`;
                    thead.innerHTML = headHtml1 + headHtml2;

                    tbody.innerHTML = '';
                    const timeSlots = Object.keys(history[camIds[0]] || {});
                    timeSlots.forEach(slot => {
                        let row = `<tr><td>${slot}</td>`;
                        camIds.forEach(id => {
                            let hData = history[id][slot] || {in:0, out:0};
                            row += `<td><span class="h-in">${hData.in}</span></td><td><span class="h-out">${hData.out}</span></td><td><span style="font-weight:bold;">${hData.in - hData.out}</span></td><td style="border-right: 2px solid #555;"></td>`;
                        });
                        row += `</tr>`;
                        tbody.innerHTML += row;
                    });
                }
            });
        }

        function loadSettings() {
            fetch('/get_cameras').then(r => r.json()).then(data => {
                currentCameras = data;
                const tbody = document.getElementById('config-table-body');
                tbody.innerHTML = '';
                Object.entries(data).forEach(([id, cam]) => {
                    // Logică pentru a afișa tipul conexiunii în tabel
                    let connInfo = cam.channel ? `Canal DVR: ${cam.channel}` : `Path: ${cam.rtsp_path || '/Streaming/Channels/101'}`;
                    
                    tbody.innerHTML += `
                        <tr>
                            <td>${id}</td><td>${cam.name}</td><td>${cam.ip}:${cam.port}</td>
                            <td style="font-size: 0.9em; color:#aaa;">${connInfo}</td>
                            <td>
                                <button class="btn-edit" onclick="editCamera('${id}')">Setări</button>
                                <button class="btn-draw" onclick="openDrawModal('${id}')">Editează Linia</button>
                                <button class="btn-delete" onclick="deleteCamera('${id}')">Șterge</button>
                            </td>
                        </tr>
                    `;
                });
            });
        }

        function saveCamera() {
            const data = {
                id: document.getElementById('f_id').value, name: document.getElementById('f_name').value,
                ip: document.getElementById('f_ip').value, port: document.getElementById('f_port').value,
                user: document.getElementById('f_user').value, password: document.getElementById('f_pass').value,
                channel: document.getElementById('f_channel').value, rtsp_path: document.getElementById('f_rtsp_path').value
            };
            if(!data.id || !data.ip) return alert('ID-ul și IP-ul sunt obligatorii!');

            fetch('/save_camera', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) })
            .then(r => r.json()).then(res => {
                if(res.success) {
                    document.getElementById('f_id').readOnly = false;
                    loadSettings();
                    setTimeout(() => { openDrawModal(res.id); }, 1000); 
                }
            });
        }

        function editCamera(id) {
            const cam = currentCameras[id];
            document.getElementById('f_id').value = id; document.getElementById('f_id').readOnly = true; 
            document.getElementById('f_name').value = cam.name; document.getElementById('f_ip').value = cam.ip;
            document.getElementById('f_port').value = cam.port; document.getElementById('f_user').value = cam.user;
            document.getElementById('f_pass').value = cam.password; 
            document.getElementById('f_channel').value = cam.channel || "";
            document.getElementById('f_rtsp_path').value = cam.rtsp_path || "";
            window.scrollTo(0, 0);
        }

        function deleteCamera(id) {
            if(confirm('Sigur vrei să ștergi camera ' + id + '?')) {
                fetch('/delete_camera', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({id: id}) })
                .then(r => r.json()).then(res => { if(res.success) loadSettings(); });
            }
        }

        function openViewModal(camId) {
            document.getElementById('view-modal').style.display = 'block';
            document.getElementById('view-stream-img').src = "/video_feed/" + camId + "?t=" + new Date().getTime();
            let camName = currentLiveDataCache[camId] ? currentLiveDataCache[camId].name : camId;
            document.getElementById('view-modal-title').innerText = "Live: " + camName;
        }

        function closeViewModal() {
            document.getElementById('view-modal').style.display = 'none';
            document.getElementById('view-stream-img').src = "";
        }

        function openDrawModal(camId) {
            activeDrawCamId = camId;
            document.getElementById('draw-modal').style.display = 'block';
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            startX = startY = endX = endY = 0;
            document.getElementById('stream-img').src = "/video_feed/" + camId + "?t=" + new Date().getTime();
        }

        function closeDrawModal() {
            document.getElementById('draw-modal').style.display = 'none';
            document.getElementById('stream-img').src = "";
            activeDrawCamId = null;
        }

        function saveDrawnLine() {
            if(!activeDrawCamId) return;
            if(startX === 0 && endX === 0) return alert("Te rugăm să desenezi o linie pe ecran!");
            const scale = 2; 
            const pt1 = [Math.round(startX * scale), Math.round(startY * scale)];
            const pt2 = [Math.round(endX * scale), Math.round(endY * scale)];

            fetch('/save_line', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id: activeDrawCamId, pt1: pt1, pt2: pt2 })
            }).then(r => r.json()).then(res => {
                if(res.success) {
                    alert('Linia a fost salvată și actualizată live!');
                    closeDrawModal();
                }
            });
        }

        setInterval(updateData, 1000);
        updateData();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

if __name__ == '__main__':
    print("=== SERVER STARTING ===")
    load_config()
    for cam_id in cameras_config:
        init_camera_structures(cam_id)
        start_camera_thread(cam_id)

    app.run(host='0.0.0.0', port=5000, debug=False)