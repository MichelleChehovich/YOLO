# =========================
# Исправленный код, адаптированный для размещения на сервере Timeweb (без Cloudflare туннеля):
# SERVER BACKEND (БЕКЕНД ДЛЯ РАЗМЕЩЕНИЯ НА СЕРВЕРЕ)
# =========================

import os
import re
import io
import json
import base64
import shutil
import zipfile
import subprocess
from pathlib import Path

# -------------------------
# 1) Install dependencies (только если запускаете в первый раз)
# -------------------------
def run_cmd(cmd, check=True, capture_output=False, text=True):
    print("RUN:", " ".join(cmd))
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=text)


# Раскомментируйте для установки зависимостей при первом запуске
# run_cmd(["python", "-m", "pip", "install", "--upgrade", "pip"])
# run_cmd(["python", "-m", "pip", "install",
#          "fastapi", "uvicorn", "python-multipart",
#          "ultralytics", "opencv-python-headless", "pillow", "numpy", "requests"])


# -------------------------
# 2) Write FastAPI app (Написать приложение FastAPI)
# -------------------------
app_code = r'''
import io
import os
import cv2
import time
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO


# -------------------------
# Config
# -------------------------
MODEL_NAME = os.getenv("YOLO_SEG_MODEL", "yolov8m-seg.pt")
DEFAULT_CLASSES = ["person", "car", "bus", "truck", "bicycle", "motorcycle"]


# Palette: BGR for OpenCV drawing
PALETTE = [
    (255, 56, 56),
    (255, 157, 151),
    (255, 112, 31),
    (255, 178, 29),
    (207, 210, 49),
    (72, 249, 10),
    (146, 204, 23),
    (61, 219, 134),
    (26, 147, 52),
    (0, 212, 187),
    (44, 153, 168),
    (0, 194, 255),
    (52, 69, 147),
    (100, 115, 255),
    (0, 24, 236),
    (132, 56, 255),
    (82, 0, 133),
    (203, 56, 255),
    (255, 149, 200),
    (255, 55, 199),
]


app = FastAPI(title="YOLOv8 Segmentation API", version="1.0.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


print(f"Loading model: {MODEL_NAME}")
model = YOLO(MODEL_NAME)
MODEL_CLASS_NAMES = model.names  # dict: id -> class_name


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_base64_png(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Failed to encode image to PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def normalize_class_list(classes_str: str):
    if not classes_str or not classes_str.strip():
        requested = DEFAULT_CLASSES
    else:
        requested = [x.strip().lower() for x in classes_str.split(",") if x.strip()]

    # unique preserve order
    seen = set()
    requested_unique = []
    for c in requested:
        if c not in seen:
            requested_unique.append(c)
            seen.add(c)

    name_to_id = {str(v).lower(): int(k) for k, v in MODEL_CLASS_NAMES.items()}
    valid_names = []
    valid_ids = []
    invalid_names = []

    for cls_name in requested_unique:
        if cls_name in name_to_id:
            valid_names.append(cls_name)
            valid_ids.append(name_to_id[cls_name])
        else:
            invalid_names.append(cls_name)

    return requested_unique, valid_names, valid_ids, invalid_names


def overlay_masks_and_collect_stats(
    image_bgr: np.ndarray,
    masks_xy,
    masks_data,
    boxes_cls,
    boxes_conf,
    class_names_map
):
    h, w = image_bgr.shape[:2]
    image_area = h * w

    overlay = image_bgr.copy()
    output = image_bgr.copy()

    # union mask per class for area stats
    class_union_masks = {}
    class_instance_counts = {}
    class_max_conf = {}

    detections = []

    if masks_data is None or len(masks_data) == 0:
        return output, {
            "found_classes": [],
            "objects_count": 0,
            "image_width": w,
            "image_height": h,
            "detections": []
        }

    for i in range(len(masks_data)):
        cls_id = int(boxes_cls[i])
        conf = float(boxes_conf[i])
        cls_name = str(class_names_map[cls_id])

        color = PALETTE[cls_id % len(PALETTE)]

        # Binary mask in original image size
        mask = masks_data[i].astype(bool)

        # Area stats: union per class
        if cls_name not in class_union_masks:
            class_union_masks[cls_name] = mask.copy()
            class_instance_counts[cls_name] = 1
            class_max_conf[cls_name] = conf
        else:
            class_union_masks[cls_name] |= mask
            class_instance_counts[cls_name] += 1
            class_max_conf[cls_name] = max(class_max_conf[cls_name], conf)

        # Draw filled mask
        overlay[mask] = (
            int(color[0]),
            int(color[1]),
            int(color[2]),
        )

        # Draw contour if polygon exists
        if masks_xy is not None and i < len(masks_xy) and masks_xy[i] is not None and len(masks_xy[i]) > 0:
            contour = np.array(masks_xy[i], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(output, [contour], isClosed=True, color=color, thickness=2)

            # label near first polygon point
            x0, y0 = contour[0][0]
            label = f"{cls_name} {conf:.2f}"
            cv2.rectangle(output, (x0, max(0, y0 - 22)), (x0 + 180, y0), color, -1)
            cv2.putText(output, label, (x0 + 4, max(12, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        detections.append({
            "class_name": cls_name,
            "confidence": round(conf, 4)
        })

    # Blend masks
    alpha = 0.45
    output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)

    # Final class stats
    found_classes = []
    for cls_name, union_mask in class_union_masks.items():
        pixels = int(union_mask.sum())
        area_percent = round(100.0 * pixels / image_area, 2)
        found_classes.append({
            "class_name": cls_name,
            "instances": int(class_instance_counts[cls_name]),
            "area_pixels": pixels,
            "area_percent": area_percent,
            "max_confidence": round(float(class_max_conf[cls_name]), 4)
        })

    # sort by area descending
    found_classes = sorted(found_classes, key=lambda x: x["area_percent"], reverse=True)

    return output, {
        "found_classes": found_classes,
        "objects_count": len(detections),
        "image_width": w,
        "image_height": h,
        "detections": detections
    }


@app.get("/")
def root():
    return {
        "message": "YOLOv8 Segmentation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "segment": "/segment (POST)"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "default_classes": DEFAULT_CLASSES,
        "available_classes_count": len(MODEL_CLASS_NAMES)
    }


@app.post("/segment")
async def segment_image(
    image: UploadFile = File(...),
    classes: str = Form("person, car, bus, truck, bicycle, motorcycle")
):
    # Засекаем общее время выполнения запроса
    request_start_time = time.time()
    
    try:
        # Засекаем время начала обработки
        process_start_time = time.time()
        
        content = await image.read()
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        image_bgr = pil_to_bgr(pil_img)

        requested_unique, valid_names, valid_ids, invalid_names = normalize_class_list(classes)

        if not valid_ids:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "No valid classes provided.",
                    "requested_classes": requested_unique,
                    "valid_classes": valid_names,
                    "invalid_classes": invalid_names,
                    "processing_time_ms": round((time.time() - process_start_time) * 1000, 2)
                }
            )

        # Засекаем время начала инференса
        inference_start_time = time.time()
        
        # Segmentation inference
        results = model.predict(
            source=image_bgr,
            classes=valid_ids,
            retina_masks=True,
            verbose=False,
            imgsz=1024,
            conf=0.25
        )
        
        # Вычисляем время инференса
        inference_time_ms = round((time.time() - inference_start_time) * 1000, 2)

        result = results[0]

        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            empty_img_b64 = bgr_to_base64_png(image_bgr)
            
            # Вычисляем общее время обработки
            total_processing_time_ms = round((time.time() - process_start_time) * 1000, 2)
            
            return {
                "success": True,
                "message": "No objects found for selected classes.",
                "requested_classes": requested_unique,
                "valid_classes": valid_names,
                "invalid_classes": invalid_names,
                "original_image_base64": empty_img_b64,
                "result_image_base64": empty_img_b64,
                "stats": {
                    "found_classes": [],
                    "objects_count": 0,
                    "image_width": image_bgr.shape[1],
                    "image_height": image_bgr.shape[0],
                    "detections": []
                },
                "processing_time": {
                    "total_ms": total_processing_time_ms,
                    "inference_ms": inference_time_ms,
                    "pre_post_ms": round(total_processing_time_ms - inference_time_ms, 2)
                }
            }

        masks_data = result.masks.data.cpu().numpy()   # [N, H, W], original size with retina_masks=True
        masks_xy = result.masks.xy                     # list of polygons
        boxes_cls = result.boxes.cls.cpu().numpy()
        boxes_conf = result.boxes.conf.cpu().numpy()

        output_bgr, stats = overlay_masks_and_collect_stats(
            image_bgr=image_bgr,
            masks_xy=masks_xy,
            masks_data=masks_data,
            boxes_cls=boxes_cls,
            boxes_conf=boxes_conf,
            class_names_map=MODEL_CLASS_NAMES
        )

        original_b64 = bgr_to_base64_png(image_bgr)
        result_b64 = bgr_to_base64_png(output_bgr)
        
        # Вычисляем общее время обработки
        total_processing_time_ms = round((time.time() - process_start_time) * 1000, 2)

        return {
            "success": True,
            "requested_classes": requested_unique,
            "valid_classes": valid_names,
            "invalid_classes": invalid_names,
            "original_image_base64": original_b64,
            "result_image_base64": result_b64,
            "stats": stats,
            "processing_time": {
                "total_ms": total_processing_time_ms,
                "inference_ms": inference_time_ms,
                "pre_post_ms": round(total_processing_time_ms - inference_time_ms, 2)
            }
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "processing_time_ms": round((time.time() - request_start_time) * 1000, 2) if 'request_start_time' in locals() else 0
            }
        )
'''

# Сохраняем приложение
Path("app.py").write_text(app_code, encoding="utf-8")

print("✅ FastAPI приложение создано: app.py")
print("\n" + "="*80)
print("ДЛЯ ЗАПУСКА НА СЕРВЕРЕ:")
print("="*80)
print("\n1. Установите зависимости (если не установлены):")
print("   pip install fastapi uvicorn python-multipart ultralytics opencv-python-headless pillow numpy requests")
print("\n2. Запустите сервер:")
print("   uvicorn app:app --host 0.0.0.0 --port 8000")
print("\n   Или для production с несколькими workers:")
print("   uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4")
print("\n3. Для запуска в фоновом режиме с systemd или supervisor:")
print("   - Создайте service файл")
print("   - Или используйте: nohup uvicorn app:app --host 0.0.0.0 --port 8000 &")
print("\n" + "="*80)
print("\n📌 ЭНДПОИНТЫ API:")
print(f"   GET  http://your-server:8000/health")
print(f"   POST http://your-server:8000/segment")
print("\n📌 ПАРАМЕТРЫ POST /segment:")
print("   image   -> файл изображения")
print("   classes -> текст, пример: person, car, bus, truck")
print("\n📌 ОТВЕТ API содержит:")
print("   - processing_time: информация о времени обработки")
print("   - stats: статистика сегментации")
print("   - original_image_base64 и result_image_base64")
print("="*80)

# Функция для вывода информации о сегментации
def print_segmentation_info(response_data):
    """
    Функция для красивого вывода информации о сегментации
    """
    print("\n" + "="*80)
    print("📊 РЕЗУЛЬТАТЫ СЕГМЕНТАЦИИ")
    print("="*80)
    
    if not response_data.get("success"):
        print(f"❌ Ошибка: {response_data.get('error', 'Неизвестная ошибка')}")
        return
    
    # Время обработки
    if "processing_time" in response_data:
        pt = response_data["processing_time"]
        print("\n⏱ ВРЕМЯ ОБРАБОТКИ:")
        print(f"   ├─ Общее время: {pt['total_ms']} мс")
        print(f"   ├─ Инференс:    {pt['inference_ms']} мс")
        print(f"   └─ Пре/пост:    {pt['pre_post_ms']} мс")
    
    # Информация о классах
    stats = response_data.get("stats", {})
    print(f"\n📦 НАЙДЕНО ОБЪЕКТОВ: {stats.get('objects_count', 0)}")
    
    # Размер изображения
    print(f"\n🖼 РАЗМЕР ИЗОБРАЖЕНИЯ: {stats.get('image_width', 0)}x{stats.get('image_height', 0)}")
    
    # Детальная информация по классам
    found_classes = stats.get("found_classes", [])
    if found_classes:
        print("\n📋 ДЕТАЛЬНАЯ ИНФОРМАЦИЯ ПО КЛАССАМ:")
        print("-" * 80)
        print(f"{'Класс':<15} {'Кол-во':<8} {'Площадь (px)':<12} {'Площадь (%)':<10} {'Max уверенность':<15}")
        print("-" * 80)
        for cls_info in found_classes:
            print(f"{cls_info['class_name']:<15} {cls_info['instances']:<8} "
                  f"{cls_info['area_pixels']:<12} {cls_info['area_percent']:<10}% "
                  f"{cls_info['max_confidence']:<15.4f}")
    else:
        print("\n📋 Объекты не найдены")
    
    # Список всех детекций
    detections = stats.get("detections", [])
    if detections:
        print("\n🔍 ВСЕ ОБНАРУЖЕННЫЕ ОБЪЕКТЫ:")
        print("-" * 60)
        for i, det in enumerate(detections, 1):
            print(f"   {i}. {det['class_name']} (уверенность: {det['confidence']:.4f})")
    
    # Информация о классах из запроса
    print(f"\n📝 ЗАПРОШЕННЫЕ КЛАССЫ: {', '.join(response_data.get('requested_classes', []))}")
    if response_data.get('invalid_classes'):
        print(f"⚠️ НЕДОСТУПНЫЕ КЛАССЫ: {', '.join(response_data['invalid_classes'])}")
    
    print("="*80 + "\n")

print("\n✅ Готово! Файл app.py создан.")
