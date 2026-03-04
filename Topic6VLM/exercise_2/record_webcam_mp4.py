import cv2
import time
from pathlib import Path

def main():
    # Output settings
    out_path = Path("webcam_recording.mp4")

    # Camera settings
    camera_index = 0  # 0 is usually the default webcam
    desired_fps = 30
    desired_width = 1280
    desired_height = 720

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    # Try to request a resolution/FPS (camera may not honor these exactly)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    cap.set(cv2.CAP_PROP_FPS, desired_fps)

    # Read back actual settings
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = desired_fps  # fallback if the camera doesn't report it

    # MP4 encoding: try H.264 first (may fail on some systems), then fall back to mp4v.
    fourcc_candidates = ["avc1", "H264", "mp4v"]
    writer = None
    for code in fourcc_candidates:
        fourcc = cv2.VideoWriter_fourcc(*code)
        w = cv2.VideoWriter(str(out_path), fourcc, float(fps), (width, height))
        if w.isOpened():
            writer = w
            print(f"Recording to: {out_path.resolve()}")
            print(f"Codec: {code}, Resolution: {width}x{height}, FPS: {fps:.2f}")
            break

    if writer is None:
        cap.release()
        raise RuntimeError(
            "Could not open VideoWriter with codecs avc1/H264/mp4v. "
            "Try installing system codecs or use .avi with XVID."
        )

    print("Press 'q' to stop recording.")
    start = time.time()
    frames_written = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed; stopping.")
                break

            # Optional: show a recording indicator + elapsed time
            elapsed = time.time() - start
            cv2.putText(
                frame,
                f"REC {elapsed:6.1f}s",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)
            frames_written += 1

            cv2.imshow("Webcam Recorder (press q to stop)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    duration = time.time() - start
    print(f"Done. Wrote {frames_written} frames in {duration:.2f}s to {out_path.resolve()}")

if __name__ == "__main__":
    main()