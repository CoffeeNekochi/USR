from threading import Thread
import time
import torch
from ultralytics import YOLO
import cv2
import platform
import sys
import numpy as np

MODEL = "yolo/yolov11n_human.pt"
SOURCE = ["test_video/test.webm"]


def get_system_info():
    """獲取系統資訊"""
    return {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "platform": platform.platform()
    }


def draw_boxes(img, boxes, person_count):
    """自定義繪製邊界框，只顯示框線"""
    img_with_boxes = img.copy()

    # 在左上角顯示人數
    cv2.putText(img_with_boxes, f'Persons: {person_count}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 繪製所有邊界框
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img_with_boxes


def thread_safe_inference(model, source):
    # 載入模型
    model = YOLO(model)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    cap = cv2.VideoCapture(source)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Video writer
    video_writer = cv2.VideoWriter("object_counting_output.avi",
                                   cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))

    # Process video
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        # 進行推論
        results = model(img, verbose=False)  # 關閉詳細輸出

        annotated_frame = []

        # 獲取人數和邊界框
        for r in results:
            # 過濾出行人檢測結果 (class 0 為 person)
            person_boxes = []
            for box in r.boxes:
                if box.cls == 0:  # person class
                    person_boxes.append(box.xyxy[0].cpu().numpy())

            person_count = len(person_boxes)

            # 使用自定義函數繪製結果
            annotated_frame = draw_boxes(img, person_boxes, person_count)

            # 輸出檢測結果
            print(f"Detected {person_count} persons")

        video_writer.write(annotated_frame)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 執行測試
    print("\n執行推論測試...")
    try:
        Thread(target=thread_safe_inference, args=(MODEL, SOURCE[0])).start()
    except Exception as e:
        print(f"測試過程中發生錯誤: {str(e)}")
