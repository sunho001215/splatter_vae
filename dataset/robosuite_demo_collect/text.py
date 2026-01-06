import ctypes, os
os.environ.setdefault("QT_QPA_PLATFORM","xcb")
ctypes.CDLL("libX11.so.6").XInitThreads()
import cv2
print("imported cv2")
cv2.namedWindow("t", cv2.WINDOW_NORMAL)
print("made window")
cv2.destroyAllWindows()
print("done")
