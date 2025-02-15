from flask import Flask, request, render_template
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('login.html')

@ app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        # get the file from the post request
        file = request.files['image']
        # save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads',file.filename)
        save_path = os.path.join(
            basepath, 'static', 'uploaded_image.jpg')
        file.save(file_path)

        fname= file.filename

        # make prediction
        image_processor = AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50",
                                                             do_resize=True,
                                                             size={"max_height": 640, "max_width": 640},
                                                             do_pad=True,
                                                             pad_size={"height": 640, "width": 640},
                                                            )
        model = torch.load("model.pt", map_location=torch.device('cpu'), weights_only=False)
        image = Image.open(file_path)
        with torch.no_grad():
            inputs = image_processor(images=[image], return_tensors="pt")
            outputs = model(**inputs.to('cpu'))
            target_sizes = torch.tensor([[image.size[1], image.size[0]]])
            results = image_processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=target_sizes)[0]
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )
        draw = ImageDraw.Draw(image)

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            x, y, x2, y2 = tuple(box)
            draw.rectangle((x, y, x2, y2), outline="red", width=1)
        image.save(save_path)
        environ = request.environ
        return render_template('login.html', fname=fname)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)




