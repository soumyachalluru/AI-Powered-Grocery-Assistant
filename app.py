def predict_item(image_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    filename = Path(image_path).stem
    print(filename)

    results = model(image_path, stream=True)

    for i, result in enumerate(results):
        image_filename = f'{output_folder}/{filename}.jpg'
        labels_filename = f'{output_folder}/{filename}.txt'
        print(labels_filename)

        result.save(filename=image_filename)

        if hasattr(result, 'boxes'):
            boxes = result.boxes
            with open(labels_filename, 'w') as f:
                for box in boxes.data:
                    class_id = int(box[5]) 
                    label = result.names[class_id]
                    bbox = tuple(box[:4].cpu().numpy()) 
                    confidence = box[4].item() 
                    f.write(f"{label} {bbox} {confidence:.2f}\n") 