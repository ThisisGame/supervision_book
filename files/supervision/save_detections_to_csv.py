import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
frames_generator = sv.get_video_frames_generator("people-walking.mp4")

with sv.CSVSink("./save_detections_to_csv.csv") as sink:
    for frame in frames_generator:

        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        sink.append(detections, {})