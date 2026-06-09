import os
import torch
import argparse
import json
from single_camera_detection.generate_yolo_datasets import generate_yolo_datasets
from single_camera_detection.yolo_functions import train_yolo_model, validate_model
from single_camera_detection.run_inference_functions import process_images_with_object_detection, process_video_with_object_detection
from utils.paths import DATA_ROOT, DATASET_ROOT, GENERATED_DATA_DIR, ensure_data_dirs


def parse_args():
    parser = argparse.ArgumentParser(description='Training and evaluation of YOLO models for single camera detection')
    
    # Create a subparser for different commands
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train single model command
    # example: python sv_main.py train_single_fold --fold fold_1 --image_size 640 --model yolo11n.pt --version v1
    train_single_fold_parser = subparsers.add_parser('train_single_fold', help='Train a single YOLO model')
    train_single_fold_parser.add_argument('--fold', type=str, required=True, choices=['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5'], 
                             help='Fold to use for training')
    train_single_fold_parser.add_argument('--image_size', type=int, required=True, help='Image size for training')
    train_single_fold_parser.add_argument('--model', type=str, required=True, help='Model string (e.g., yolo11x.pt)')
    train_single_fold_parser.add_argument('--version', type=str, default='v1', help='Version string')
    
    # Train all models command
    train_all_parser = subparsers.add_parser('train_all', help='Train multiple YOLO models')
    train_all_parser.add_argument('--version', type=str, default='v1')


    test_all_models_parser = subparsers.add_parser('test_all_models', help='Test all YOLO models')
    test_all_models_parser.add_argument('--version', type=str, default='v1')
    
    # Validate models command
    val_parser = subparsers.add_parser('validate', help='Validate trained models')
    val_parser.add_argument('--fold', type=str, required=True, choices=['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5'], 
                           help='Fold to use for validation')
    val_parser.add_argument('--image_size', type=int, default=None, help='Specific image size to validate')
    val_parser.add_argument('--model', type=str, default=None, help='Specific model to validate')

    # Train final model command
    train_final_parser = subparsers.add_parser('train_final_model', help='Train final YOLO model')
    train_final_parser.add_argument('--model', type=str, default='yolov8x.pt', help='Model string (e.g., yolo11x.pt)')
    train_final_parser.add_argument('--image_size', type=int, default=1920, help='Image size for training')
    train_final_parser.add_argument('--version', type=str, default='v1', help='Version string')
    
    # Gather detections command
    gather_detections_parser = subparsers.add_parser('gather_detections', help='Gather detections from a model')
    gather_detections_parser.add_argument('--image_size', type=int, default=1920, help='Image size for training')

    gather_detections_video_parser = subparsers.add_parser('gather_detections_video', help='Gather detections from a model')
    gather_detections_video_parser.add_argument('--image_size', type=int, default=1920, help='Image size for training')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)

    ensure_data_dirs()
    mvbrotrack_root_path = DATASET_ROOT

    if not os.path.exists(mvbrotrack_root_path):
        print(f"MVBroTrack root path not found at {mvbrotrack_root_path}")
        user_input = input("Do you want to download MVBroTrack? (yes/no): ").strip().lower()
        if user_input in ['yes', 'y']:
            zip_path = os.path.join(DATA_ROOT, 'MVBroTrack_v2.zip')
            print("Downloading MVBroTrack from https://datasets.ilabt.imec.be/broilertracking/MVBroTrack_v2.zip")
            os.system(f'wget https://datasets.ilabt.imec.be/broilertracking/MVBroTrack_v2.zip -O {zip_path}')
            os.system(f'unzip {zip_path} -d {DATA_ROOT}')
            os.system(f'rm {zip_path}')
        else:
            print("Download cancelled. Exiting.")
            exit(1)


    # setting up paths 
    root_output_path = GENERATED_DATA_DIR
    os.makedirs(root_output_path, exist_ok=True)

    single_view_output_path = os.path.join(root_output_path, 'Single-View-Detection')
    os.makedirs(single_view_output_path, exist_ok=True)

    models_dir = os.path.join(single_view_output_path, 'trained_runs')
    os.makedirs(models_dir, exist_ok=True)

    svbrotrack_yolo_data_path = os.path.join(single_view_output_path, 'yolo_data')

    if not os.path.exists(svbrotrack_yolo_data_path) and args.command != 'gather_detections' and args.command != 'gather_detections_video':
        print(f"YOLO data path not found at {svbrotrack_yolo_data_path}")
        print("Generating YOLO datasets...")
        generate_yolo_datasets(os.path.join(mvbrotrack_root_path, 'Single-View-Detection'), svbrotrack_yolo_data_path)

    if args.command == 'train_single_fold':
        yaml_path = os.path.join(svbrotrack_yolo_data_path, 'datasets', args.fold + '.yaml')
        if not os.path.exists(yaml_path):
            print(f"YAML file not found: {yaml_path}")
            exit(1)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        print(f"Training single model: {args.model} with image size {args.image_size} on {args.fold}")
        project_name = args.model.split('.')[0] + '_' + str(args.image_size) + '_' + args.fold + '_' + args.version
        train_name_path = os.path.join(models_dir, project_name)
        train_yolo_model(yaml_path, args.image_size, args.model, train_name_path)
    
    elif args.command == 'train_all':
        print("Training all model configurations")
        model_strings = ['yolo11x.pt', 'yolo11m.pt', 'yolo11n.pt'] 
        # model_strings = ['yolov8x.pt', 'yolov8m.pt', 'yolov8n.pt'] 
        image_sizes = [1920, 1280, 960, 640, 320]
        folds = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        for image_size in image_sizes:
            for model_string in model_strings:
                for fold in folds:
                    yaml_path = os.path.join(svbrotrack_yolo_data_path, 'datasets', fold + '.yaml')
                    if not os.path.exists(yaml_path):
                        print(f"YAML file not found: {yaml_path}. Skipping...")
                        continue
                    
                    print(f"Training: {model_string} with image size {image_size} on {fold}")
                    project_name = model_string.split('.')[0] + '_' + str(image_size) + '_' + fold + '_' + args.version
                    train_name_path = os.path.join(models_dir, project_name)

                    if os.path.exists(train_name_path):
                        print(f"Model already exists: {train_name_path}")
                        continue
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                    print("Cleared cache")
                    print("Training model...")
                    train_yolo_model(yaml_path, image_size, model_string, train_name_path)
            
    elif args.command == 'test_all_models':
        root_to_all_yaml = os.path.join(svbrotrack_yolo_data_path, 'datasets')
        version_number = args.version

        root_to_all_test_results = os.path.join(single_view_output_path, 'test_results_' + version_number)
        json_save_name = os.path.join(single_view_output_path, 'results_dict_' + version_number + '.json')

        os.makedirs(root_to_all_test_results, exist_ok=True)
        all_models = os.listdir(models_dir)
        print(f'Evaluating {len(all_models)} models for version {version_number}')

        results_dict = {}

        for project_name in all_models:
            print(f"Validating model: {project_name}")
            model_name = project_name.split('_')[0]
            fold_number = project_name.split('_')[3]
            img_size = project_name.split('_')[1]

            version = project_name.split('_')[-1]
            if version != version_number:
                continue

            model_path = os.path.join(models_dir, project_name, 'train', 'weights', 'best.pt')
            print(f"Model path: {model_path}")

            yaml_path = os.path.join(root_to_all_yaml, 'fold_'+fold_number+'.yaml')
            print(f"YAML path: {yaml_path}")
            project_name = model_name + '_' + img_size + '_fold_' + fold_number + '_v' + str(version)

            project_path = os.path.join(root_to_all_test_results, project_name)

            metrics = validate_model(model_path, yaml_path, img_size, project_path)

            print(f"Metrics: {metrics.results_dict}")
            results_dict[project_name] = metrics.results_dict

            for phase in ['starter', 'grower', 'finisher']:
                yaml_path = os.path.join(root_to_all_yaml, 'fold_'+fold_number+'_'+phase+'.yaml')

                print(f"YAML path: {yaml_path}")
                project_name = model_name + '_' + img_size + '_fold_' + fold_number + '_' + phase + '_' + str(version) 


                project_path = os.path.join(root_to_all_test_results, project_name)
                metrics = validate_model(model_path, yaml_path, img_size, project_path)

                print(f"Metrics: {metrics.results_dict}")
                results_dict[project_name] = metrics.results_dict

                with open(json_save_name, 'w') as f:
                    json.dump(results_dict, f)
                    
    elif args.command == 'train_final_model':
        print("Training final model")
        model_string = args.model
        image_size = args.image_size

        yaml_path = os.path.join(svbrotrack_yolo_data_path, 'datasets', 'final_test.yaml')
        train_name_path = os.path.join(models_dir, f'{model_string.split(".")[0]}_{image_size}_final_{args.version}')
        os.makedirs(train_name_path, exist_ok=True)

        print("Training model...")
        train_yolo_model(yaml_path, image_size, model_string, train_name_path)

    elif args.command == 'gather_detections':
        images_folder = os.path.join(mvbrotrack_root_path, 'Multi-View-Detection', 'images')
        if not os.path.exists(images_folder):
            print(f"Images folder not found: {images_folder}")
            exit(1)

        model_string = 'yolo11x_1920.pt' # also has yolo8x_1920.pt
        model_path = os.path.join(mvbrotrack_root_path,'Single-View-Detection', 'trained_models', model_string)
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            exit(1)

        output_folder = os.path.join(root_output_path, 'Multi-View-Detection', model_string.split('.')[0])
        os.makedirs(output_folder, exist_ok=True)

        process_images_with_object_detection(images_folder, output_folder, model_path, args.image_size)

    elif args.command == 'gather_detections_video':
        model_string = 'yolo11x_1920.pt' # also has yolo8x_1920.pt
        model_path = os.path.join(mvbrotrack_root_path,'Single-View-Detection', 'trained_models', model_string)
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            exit(1)

        tracking_root = os.path.join(mvbrotrack_root_path, 'Multi-View-Tracking')
        if not os.path.exists(tracking_root):
            print(f"Multi-View-Tracking root not found: {tracking_root}")
            exit(1)

        for sample_number in range(1, 7): # 6 * 4 = 24 videos
            video_folder = os.path.join(tracking_root, f'sample_{sample_number}', 'videos')
            if not os.path.exists(video_folder):
                print(f"Video folder not found: {video_folder}")
                exit(1)

            output_folder = os.path.join(root_output_path, 'Multi-View-Tracking', f'sample_{sample_number}', model_string.split('.')[0])
            os.makedirs(output_folder, exist_ok=True)

            all_videos = os.listdir(video_folder)
            for video in all_videos:
                video_path = os.path.join(video_folder, video)
                output_folder_video = os.path.join(output_folder, video.split('.')[0])
                os.makedirs(output_folder_video, exist_ok=True)

                process_video_with_object_detection(video_path, output_folder_video, model_path, args.image_size)

    else:
        print("Please specify a command: train_single_fold, train_all, train_final_model, gather_detections, or gather_detections_video")
        print("Use --help for more information")