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

    def split_text_line(line):
        line = line.strip()

        if 'gm' in line:
            unit = 'gm'
        elif 'ml' in line:
            unit = 'ml'
        else:
            return None 
        
        quantity_end_index = line.find(unit) + 2 
        product_name = line[:quantity_end_index].rsplit(' ', 1)[0]
        quantity = line[quantity_end_index-4:quantity_end_index].strip()
        rest = line[quantity_end_index+1:].strip()

        return product_name, quantity, rest
    file_path = labels_filename

    with open(file_path, 'r') as file:
        for line in file:
            product_name, quantity, rest = split_text_line(line)
            print("Product Name:", product_name)
            print("Quantity:", quantity)
            print("The Rest:", rest) 
            return product_name, quantity