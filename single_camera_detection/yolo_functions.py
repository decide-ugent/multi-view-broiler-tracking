from ultralytics import YOLO

def train_yolo_model(yaml_file, img_size, model_string, train_name_path):
    model = YOLO(model_string)  # load a pretrained model
    model.train(
        data=yaml_file,
        imgsz=img_size,
        # use auto batch size
        # batch=-1,
        batch=0.5,
        # batch=1,
        workers=2,
        # patience=50,
        epochs=200,
        project=train_name_path,
        # device=[0,1]
    )

def validate_model(model_path, yaml_path, img_size, project_path):
    print(f"Validating model: {model_path}")

    model = YOLO(model_path)
    metrics = model.val(data=yaml_path, 
                        batch=1,
                        imgsz=img_size,
                        save_json=True, 
                        # save_hybrid=True, 
                        conf=0.4,
                        project=project_path,
                        split='test',
                        )
    return metrics