import numpy as np
import cv2
import segmentation_models_pytorch as smp
import torch
import torchvision.transforms as tvt
import matplotlib.pyplot as plt
import argparse
import os
import sys


def main():
    # Gets the user parameters
    parser = argparse.ArgumentParser(description='script to segment the polar body')
    parser.add_argument('--input', help='path/to/input <image file>', type=str, required=True)
    parser.add_argument('--output', help='path/to/output <text file>', type=str, required=True)
    args = parser.parse_args()

    img_name = args.input
    txt_name = args.output
    base_name = os.path.basename(img_name).split('.')[0]

    out_dir = os.path.dirname(txt_name)
    if out_dir == '':
        out_dir = './'
        txt_name = os.path.join(out_dir, txt_name)

    # Sets the model weights
    det_weights_path = 'weights/oocyte_det.pt'
    seg_weights_path = 'weights/polar_body_seg.pt'

    # Checks the different src/dst paths
    if not os.path.exists(det_weights_path):
        sys.exit('Error: path to "%s" does not exist' % det_weights_path)

    if not os.path.exists(seg_weights_path):
        sys.exit('Error: path to "%s" does not exist' % seg_weights_path)

    if not os.path.exists(img_name):
        sys.exit('Error: path to "%s" does not exist' % img_name)

    if not os.path.exists(os.path.dirname(txt_name)):
        sys.exit('Error: path to "%s" does not exist' % txt_name)

    try:
        # Sets a device
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        # Creates the detection model
        print('Detecting object...')
        det_model = torch.hub.load('.',
                                   'custom',
                                   path=det_weights_path,
                                   force_reload=True,
                                   source='local')

        # Initializes the YOLOv5 model
        det_model.compute_iou = 0.6  # IoU threshold
        det_model.conf = 0.6  # Confidence threshold
        det_model.max_det = 1  # Max number of detections

        # Performs object detection with YOLOv5
        results = det_model(img_name)
        detections = results.xyxy[0].detach().cpu().numpy()

        # Checks if there is detections:
        if len(detections) > 0:
            box = detections[0][:4]
            x, y, w, h = round(box[0]), round(box[1]), round(box[2] - box[0]), round(box[3] - box[1])
        else:
            sys.exit('No oocyte detection')
        print('Done\n')

        # Creates the segmentation model
        print('Segmenting object...')

        seg_model = smp.Unet(
            encoder_name=ENCODER_NAME,
            encoder_weights='imagenet',
            in_channels=N_CHANNELS,
            classes=N_CHANNELS,
        )

        # Puts the model in a device (CPU or GPU)
        seg_model = seg_model.to(device)

        # Loads the trained model
        seg_model.load_state_dict(torch.load(seg_weights_path))  # Load trained model

        # Sets the model to evaluation mode
        seg_model.eval()
        print('Done')

        # Sets the graph parameters
        color = np.array([255, 0, 0], dtype='uint8')
        alpha = 0.4
        title = 'Model: %s  Encoder: %s \nInput image format: %s' % (MODEL_NAME, ENCODER_NAME, IMG_FORMAT)

        # Reads the input image
        color_img = cv2.imread(img_name)

        # Creates a CLAHE object for image contrast enhancing
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Applies the CLAHE algorithm to a grayscale image
        img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        img = clahe.apply(img).astype(np.float32)

        # Crops the predicted mask
        crop_img = img[y:y + h, x:x + w]
        height_orig = crop_img.shape[0]
        width_orig = crop_img.shape[1]

        # Composes the image transformations
        transform_img = tvt.Compose([tvt.ToPILImage(),
                                     tvt.Resize((IMG_HEIGHT, IMG_WIDTH), tvt.InterpolationMode.BILINEAR),
                                     tvt.PILToTensor()])

        # Applies the transformations to the cropped image
        crop_img = transform_img(crop_img)

        # Puts the transformed image in a device
        crop_img = torch.autograd.Variable(crop_img, requires_grad=False).to(device).unsqueeze(0)

        # Performs object segmentation
        with torch.no_grad():
            prd = seg_model(crop_img.float())

        # Resizes the predicted mask
        prd = tvt.Resize((height_orig, width_orig), tvt.InterpolationMode.NEAREST)(prd[0])

        # Puts the predicted mask in a numpy array
        seg = prd.data.cpu().detach().numpy().squeeze(0)

        # Converts the mask values to integers
        seg = (seg > 0.5).astype(np.uint8)

        # Evaluates the oocyte maturity
        pb = np.count_nonzero(seg)
        if pb > MATURITY_THR:
            is_mature = '1'
        else:
            is_mature = '0'

        # Saves the results (0=Non-maturity, 1=Maturity) to a text file
        output = open(txt_name, "w")
        output.write(is_mature)
        output.close()

        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        masked_img = np.where(seg[..., None], color, color_img[y:y + h, x:x + w])

        # Creates the superimposed image
        out = cv2.addWeighted(color_img[y:y + h, x:x + w], 1 - alpha, masked_img, alpha, 0)

        # Plots the crop/mask images
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[10, 8])
        fig.suptitle(title)
        axes[0].axis('off')
        axes[0].set_title('Crop')
        axes[0].imshow(color_img[y:y + h, x:x + w])
        axes[1].axis('off')
        axes[1].set_title('Crop + predicted mask')
        axes[1].imshow(out)
        fig.tight_layout()
        plt.savefig(os.path.join(out_dir, base_name + '_result.jpg'))
        plt.close('all')

    except Exception as e:
        print('Exception: %s' % str(e))


if __name__ == '__main__':
    # Main parameters
    MODEL_NAME = 'Unet'
    ENCODER_NAME = 'resnet101'
    IMG_FORMAT = 'grayscale'
    N_CHANNELS = 1
    N_CLASSES = 1
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    MATURITY_THR = 10

    # Call to main function
    main()
