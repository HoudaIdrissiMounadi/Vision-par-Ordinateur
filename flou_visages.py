"""Mini-projet : floutage automatique des visages (anonymisation).

Usage :
    python flou_visages.py image  people.jpg   sortie.jpg
    python flou_visages.py video  video.mp4    sortie.mp4
    python flou_visages.py webcam                      (touche 'q' pour quitter)
"""
import sys

import cv2
import dlib

detector = dlib.get_frontal_face_detector()


def flouter_visages(frame, noyau=(51, 51)):
    """Détecte les visages et applique un flou gaussien sur chaque zone."""
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = frame.shape[:2]
    for f in detector(gris):
        x1, y1 = max(f.left(), 0), max(f.top(), 0)
        x2, y2 = min(f.right(), w), min(f.bottom(), h)
        zone = frame[y1:y2, x1:x2]
        if zone.size:
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(zone, noyau, 0)
    return frame


def traiter_image(entree, sortie):
    img = cv2.imread(entree)
    if img is None:
        sys.exit(f"Image introuvable : {entree}")
    cv2.imwrite(sortie, flouter_visages(img))
    print("Image enregistrée :", sortie)


def traiter_video(source, sortie=None, afficher=False):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Impossible d'ouvrir la source : {source}")
    out = None
    if sortie:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        out = cv2.VideoWriter(sortie, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = flouter_visages(frame)
        if out:
            out.write(frame)
        if afficher:
            cv2.imshow("Floutage des visages (q pour quitter)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if out:
        out.release()
    if afficher:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    mode = sys.argv[1]
    if mode == "image" and len(sys.argv) == 4:
        traiter_image(sys.argv[2], sys.argv[3])
    elif mode == "video" and len(sys.argv) == 4:
        traiter_video(sys.argv[2], sys.argv[3])
    elif mode == "webcam":
        traiter_video(0, afficher=True)
    else:
        sys.exit(__doc__)
