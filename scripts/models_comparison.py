import face_detection_models as fm
import os

photos = os.listdir("face_comparison_dataset")
for i in photos:
    print(i)
    fm.detection_haar(f"face_comparison_dataset/{i}",scale_factor=1.05,model_label=True)
    fm.detection_haar(f"face_comparison_dataset/{i}",scale_factor=1.10,model_label=True)
    fm.detection_haar(f"face_comparison_dataset/{i}",scale_factor=1.20,model_label=True)
    fm.detection_mtcnn(f"face_comparison_dataset/{i}",model_label=True)
    fm.detection_dnn(f"face_comparison_dataset/{i}",min_confidence_score=0.2,model_label=True)
    fm.detection_dnn(f"face_comparison_dataset/{i}",min_confidence_score=0.4,model_label=True)
    fm.detection_dnn(f"face_comparison_dataset/{i}",min_confidence_score=0.6,model_label=True)