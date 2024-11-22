import torch
import torchvision
import cv2
from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    print("Torch version:", torch.__version__)  # Expected output: 2.2.1+cu118
    print("Torchvision version:", torchvision.__version__)  # Expected output: 0.17.1+cu118
    print("OpenCV version:", cv2.__version__)  # Should output OpenCV version, e.g., 4.6.0 or higher
    print("CUDA available:", torch.cuda.is_available())  # Should return True if CUDA is configured correctly

    # Load a model
    model = YOLO("yolo11m.pt")  # Make sure this model path is correct

    # Train the model with `workers=0` to disable multiprocessing
    results = model.train(data="config.yaml", epochs=1, workers=0)  # Specify data config and other training parameters

if __name__ == '__main__':
    freeze_support()  # Required on Windows to safely handle multiprocessing
    main()
