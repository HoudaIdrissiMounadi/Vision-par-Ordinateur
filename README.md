# TP 3 : Vision par ordinateur

Module Technologies IA, ENSA Fès, ILIA2.

## Contenu du dépôt

| Fichier | Description |
|---|---|
| `TP3_Vision_par_ordinateur.ipynb` | Notebook avec les exercices 1 à 7 (code complet) et les réponses aux questions (cellules Markdown) |
| `flou_visages.py` | Mini-projet : floutage automatique des visages |
| `captures/` | Captures d'écran du résultat de chaque exercice |

## Exécution du notebook

Le notebook est conçu pour **Google Colab** (GPU gratuit recommandé) :

1. Ouvrir le notebook dans [Google Colab](https://colab.research.google.com/) (*Fichier > Importer un notebook*, ou via GitHub).
2. Activer le GPU : *Exécution > Modifier le type d'exécution > GPU*.
3. Exécuter les cellules dans l'ordre et téléverser les images demandées : `image1.jpg`, `image_objects.jpg`, `people.jpg`, `ref.jpg`, `test.jpg`.

Detectron2 s'installe difficilement sous Windows, d'où l'usage de Colab pour les parties 3 et 7.

## Mini-projet : floutage automatique des visages

**Objectif :** détecter les visages (dlib) puis appliquer `cv2.GaussianBlur` sur chaque zone pour anonymiser une image, une vidéo ou le flux de la webcam (conformité vie privée).

**Installation :**

```bash
pip install opencv-python dlib-bin
```

**Utilisation :**

```bash
python flou_visages.py image  people.jpg sortie.jpg
python flou_visages.py video  video.mp4  sortie.mp4
python flou_visages.py webcam              # 'q' pour quitter
```

## Ressources

- OpenCV : https://opencv.org/
- Detectron2 : https://github.com/facebookresearch/detectron2
- Dlib : http://dlib.net/
- face_recognition : https://github.com/ageitgey/face_recognition
- Ultralytics YOLO : https://docs.ultralytics.com/
