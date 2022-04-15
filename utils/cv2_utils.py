import cv2


def save_frames(path, frames):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    size = [frames.shape[2], frames.shape[1]]
    fps = 25
    vw = cv2.VideoWriter(path, fourcc, fps, size, isColor = True)
    for frame in frames:
        vw.write(frame)
    vw.release()