import sys
import argparse
import logging
from pathlib import Path
import cv2

# Set up paths so we can import core
_app_dir = Path(__file__).resolve().parent
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

from core.detector import FishDetector, draw_boxes_on_frame
from core.counter import DetectionLineCounter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run the Detection Line Counter on a video.")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--out", type=str, default="output.mp4", help="Path to output video file")
    
    # YOLO params
    parser.add_argument("--conf", type=float, default=0.1, help="YOLO confidence threshold")
    
    # Counter params
    parser.add_argument("--line_y", type=int, default=540, help="Counting line Y coordinate (default 540 for 1080p)")
    parser.add_argument("--band_px", type=int, default=90, help="Band +/- size around line_y (default 90)")
    parser.add_argument("--x_tolerance", type=int, default=150, help="Max expected X displacement (default 150)")
    parser.add_argument("--y_tolerance", type=int, default=230, help="Max expected Y displacement (default 230)")
    parser.add_argument("--max_frame_gap", type=int, default=5, help="Max frames between orphan detections (default 5)")
    parser.add_argument("--conf_thresh", type=float, default=0.15, help="Counter confidence threshold (default 0.15)")
    
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    if not video_path.exists():
        logger.error(f"Input video not found: {video_path}")
        sys.exit(1)
        
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        sys.exit(1)
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))
    
    detector = FishDetector()
    if not detector.is_loaded:
        logger.error("Failed to load YOLO model.")
        sys.exit(1)
        
    counter = DetectionLineCounter(
        line_y=args.line_y,
        band_px=args.band_px,
        x_tolerance=args.x_tolerance,
        y_tolerance=args.y_tolerance,
        max_frame_gap=args.max_frame_gap,
        conf_thresh=args.conf_thresh
    )
    
    logger.info(f"Starting processing: {video_path.name} ({width}x{height} @ {fps}fps)")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = detector.predict(frame, confidence=args.conf)
        
        detections = []
        boxes_list = []
        confs_list = []
        
        if len(results) > 0:
            res = results[0]
            if len(res.boxes) > 0:
                boxes_tensor = res.boxes.xyxy.cpu().numpy()
                confs_tensor = res.boxes.conf.cpu().numpy()
                
                for i, box in enumerate(boxes_tensor):
                    x1, y1, x2, y2 = box
                    conf = confs_tensor[i]
                    
                    w = x2 - x1
                    h = y2 - y1
                    
                    detections.append((x1, y1, w, h, float(conf)))
                    
                    boxes_list.append(box.tolist())
                    confs_list.append(float(conf))
                    
        # Update counter
        counter.process_frame(frame_idx, detections)
        
        # Annotate frame
        annotated_frame = counter.annotate_frame(frame)
        if len(boxes_list) > 0:
             annotated_frame = draw_boxes_on_frame(annotated_frame, boxes_list, confs_list)
             
        out_writer.write(annotated_frame)
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            logger.info(f"Processed {frame_idx}/{total_frames} frames. Current count: {counter.get_count()}")
            
    counter.flush(frame_idx)
    cap.release()
    out_writer.release()
    
    logger.info(f"Finished processing. Total Count: {counter.get_count()}")
    logger.info(f"Output saved to {args.out}")

if __name__ == '__main__':
    main()
