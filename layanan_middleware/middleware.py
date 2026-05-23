import os
import sys
import traceback
import cv2
import numpy as np
import torch
import pickle
import tempfile
import uuid
import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
import uvicorn
from multiprocessing import Manager, Process
from utils import (
    InferenceDataPreprocessor, 
    STGCN, 
    build_adjacency_matrix, 
    seq_to_tensor, 
    draw_keypoints_and_connections,
    IN_CHANNELS,
    NUM_NODES,
    SEQ_LENGTH,
)

# Initialize API
app = FastAPI(
    title="Ling Tien Kung HAR Middleware API",
    description="Middleware API to process video in the background",
    version="3.0.0"
)

# --- Konfigurasi dan Variabel Global ---
# Ganti path ini sesuai dengan lokasi file model dan data Anda
YOLO_MODEL_PATH = './models/yolov8n-pose.pt'
MODEL_PATH = './models/stgcn_model_cpu.pth'
CLASS_MAPPING_PATH = './models/class_mapping.pkl'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Variabel global untuk model, preprocessor, dan mapping
# NOTE: job_queue sekarang akan disimpan di app.state
stgcn_model = None
data_preprocessor = None
class_mapping = None
class_mapping_rev = None

# --- Event Handler Startup ---
@app.on_event("startup")
def startup_event():
    """
    Fungsi ini dijalankan sekali untuk setiap proses worker Uvicorn.
    Ini adalah tempat yang aman untuk menginisialisasi shared state.
    """
    manager = Manager()
    app.state.job_queue = manager.dict()
    print(f"Worker {os.getpid()} - Multiprocessing Manager dan app.state.job_queue berhasil diinisialisasi.")

# --- Fungsi untuk Memuat Model (Agar bisa diakses oleh semua proses) ---
def load_models():
    """Fungsi ini memuat semua model dan preprocessor."""
    try:
        # Load class mapping
        with open(CLASS_MAPPING_PATH, 'rb') as f:
            class_mapping_loaded = pickle.load(f)
        
        class_mapping_rev_loaded = {v: k for k, v in class_mapping_loaded.items()}
        NUM_CLASSES = len(class_mapping_loaded)

        # Build adjacency matrix
        A = build_adjacency_matrix(NUM_NODES)

        # Initialize and load STGCN model
        stgcn_model_loaded = STGCN(IN_CHANNELS, NUM_NODES, NUM_CLASSES, SEQ_LENGTH, A)
        stgcn_model_loaded.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        stgcn_model_loaded.to(DEVICE)
        stgcn_model_loaded.eval()
        # Initialize data preprocessor
        data_preprocessor_loaded = InferenceDataPreprocessor(yolo_model_path=YOLO_MODEL_PATH, device=DEVICE)

        return stgcn_model_loaded, data_preprocessor_loaded, class_mapping_rev_loaded
    except Exception as e:
        print(f"Error saat memuat model: {e}\n{traceback.format_exc()}")
        return None, None, None

# --- Fungsi untuk Menghapus File ---
def cleanup_file(file_path: str):
    """
    Fungsi utilitas untuk menghapus file dan mencatat prosesnya.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Berhasil menghapus file: {file_path}")
        else:
            print(f"Gagal menghapus file: {file_path} (tidak ditemukan)")
    except Exception as e:
        print(f"Error saat menghapus file {file_path}: {e}\n{traceback.format_exc()}")

# --- Fungsi Pemrosesan Video di Background ---
def process_video(job_id: str, input_path: str, output_path: str, shared_job_queue):
    """
    Fungsi ini menjalankan logika pemrosesan video di proses terpisah.
    shared_job_queue diteruskan secara eksplisit dari proses induk.
    """
    shared_job_queue[job_id] = {"status": "processing", "output_path": None, "error": None}

    try:
        # Muat model di dalam proses ini
        stgcn_model_local, data_preprocessor_local, class_mapping_rev_local = load_models()
        if stgcn_model_local is None:
             raise RuntimeError("Gagal memuat model di proses background.")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("Gagal membuka file video yang diunggah.")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # ===============================================================
        #DAERAH PERUBAHAN!
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        fourcc = cv2.VideoWriter_fourcc(*'avc1') 
        # ===============================================================
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            raise RuntimeError("Gagal menginisialisasi VideoWriter.")

        keypoints_buffer_normalized = []
        keypoints_buffer_raw = []
        predicted_label = "Processing..."
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            normalized_kpts, raw_kpts = data_preprocessor_local.extract_keypoints(frame)

            if normalized_kpts is not None:
                keypoints_buffer_normalized.append(normalized_kpts)
                keypoints_buffer_raw.append(raw_kpts)
            else:
                keypoints_buffer_normalized.append(np.zeros((NUM_NODES, IN_CHANNELS)))
                keypoints_buffer_raw.append(np.zeros((NUM_NODES, IN_CHANNELS)))

            if len(keypoints_buffer_normalized) >= SEQ_LENGTH:
                current_sequence_normalized = np.array(keypoints_buffer_normalized[-SEQ_LENGTH:])
                sequence_tensor = seq_to_tensor(current_sequence_normalized).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    outputs = stgcn_model_local(sequence_tensor)
                    _, predicted_id = torch.max(outputs.data, 1)
                    predicted_label = class_mapping_rev_local.get(predicted_id.item(), "Unknown")

            current_raw_kpts = keypoints_buffer_raw[-1] if keypoints_buffer_raw else np.zeros((NUM_NODES, IN_CHANNELS))
            annotated_frame = draw_keypoints_and_connections(frame.copy(), current_raw_kpts, predicted_label)
            out.write(annotated_frame)

        cap.release()
        out.release()
        
        if frame_count == 0:
            raise RuntimeError("Video tidak mengandung frame atau tidak dapat diproses.")
        
        # Perbarui status di shared_job_queue
        shared_job_queue[job_id] = {"status": "completed", "output_path": output_path, "error": None}
        print(f"Job {job_id} selesai. Video output di: {output_path}")

    except Exception as e:
        error_message = f"Terjadi kesalahan saat memproses video: {e}"
        print(f"Error pada job {job_id}: {error_message}\n{traceback.format_exc()}")
        
        # Perbarui status di shared_job_queue
        shared_job_queue[job_id] = {"status": "failed", "output_path": None, "error": error_message}
    finally:
        # Selalu hapus file input sementara
        cleanup_file(input_path)
        # Hapus juga file output jika terjadi error di tengah jalan
        if shared_job_queue.get(job_id, {}).get("status") == "failed":
            cleanup_file(output_path)

# --- Endpoint Asinkron untuk Mengunggah Video ---
@app.post("/submit_video/")
async def submit_video(request: Request, video_file: UploadFile = File(...)):
    """Menerima video, memulai pemrosesan di background, dan mengembalikan job ID."""
    job_id = str(uuid.uuid4())
    temp_input_video_path = tempfile.mktemp(suffix=".mp4")
    temp_output_video_path = tempfile.mktemp(suffix=".mp4")
    
    # Simpan file yang diunggah ke path sementara
    try:
        with open(temp_input_video_path, "wb") as f:
            f.write(await video_file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan file: {e}")
        
    print(f"Menerima job {job_id} untuk video: {video_file.filename}")

    #=======================================
    # DAERAH PERBAIKAN/PENAMBAHAN!
    #=======================================

    request.app.state.job_queue[job_id] = {"status": "pending", "output_path": None, "error": None}
    
    # Menjalankan pemrosesan video di proses terpisah agar tidak memblokir server
    # Meneruskan `app.state.job_queue` secara eksplisit
    p = Process(target=process_video, args=(job_id, temp_input_video_path, temp_output_video_path, request.app.state.job_queue))
    p.start()
    
    return JSONResponse(content={"job_id": job_id})

# --- Endpoint Asinkron untuk Mendapatkan Hasil ---
@app.get("/get_result/{job_id}")
async def get_result(request: Request, job_id: str):
    """
    Memeriksa status job pemrosesan.
    Jika selesai, mengembalikan video yang telah diproses.
    """
    job = request.app.state.job_queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job ID tidak ditemukan.")

    status = job["status"]

    if status == "completed":
        output_path = job["output_path"]
        if os.path.exists(output_path):
            return FileResponse(
                path=output_path,
                media_type="video/mp4",
                filename=f"annotated_video_{job_id}.mp4",
                background=BackgroundTask(lambda: cleanup_file(output_path))
            )
        else:
            # File tidak ditemukan, mungkin sudah dihapus
            raise HTTPException(status_code=500, detail="File output tidak ditemukan.")
    
    elif status == "failed":
        error_message = job["error"]
        return JSONResponse(content={"status": "failed", "error": error_message})
    
    else: # "pending" atau "processing"
        return JSONResponse(content={"status": status})

# --- Jalankan Uvicorn ---
if __name__ == "__main__":
    # Event handler `startup` akan menginisialisasi `app.state.job_queue`
    uvicorn.run(app, host="0.0.0.0", port=8000)
