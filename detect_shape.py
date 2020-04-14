from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import os
import sys
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def detect(image_root, model_path, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    checkpoint = torch.load(model_path)
    start_epoch = checkpoint['epoch'] + 1
    print(model_path)
    print('Loading checkpoint from epoch %d.\n' % start_epoch)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    print(os.listdir(image_root))
    for img_name in os.listdir(image_root):
        if not img_name.endswith('.jpg') and not img_name.endswith('.png'):
            continue

        if img_name.endswith('.jpg'):
            suffix = '.jpg'
        else:
            suffix = '.png'

        original_image = Image.open(os.path.join(image_root, img_name), mode='r')
        original_image = original_image.convert('RGB')

        # Transform
        image = normalize(to_tensor(resize(original_image)))

        # Move to default device
        image = image.to(device)

        # Forward prop.
        predicted_locs, predicted_scores = model(image.unsqueeze(0))

        # Detect objects in SSD output
        det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                                 max_overlap=max_overlap, top_k=top_k)

        # Move detections to the CPU
        det_boxes = det_boxes[0].to('cpu')

        # Transform to original image dimensions
        original_dims = torch.FloatTensor(
            [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
        det_boxes = det_boxes * original_dims

        # Decode class integer labels
        det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

        # If no objects found, the detected labels will be set to ['0.']
        # i.e. ['background'] in SSD300.detect_objects() in model.py
        if det_labels == ['background']:
            # Just return original image
            print('No object is detected in this image:', img_name)
            # return original_image
            continue

        # Annotate
        annotated_image = cv2.imread(os.path.join(image_root, img_name))

        # # Suppress specific classes, if needed
        # print('Returned det boxes: ', det_boxes.size())
        # print('Labels: ', det_labels)
        # print('Scores: ', det_scores)
        # for i in range(det_boxes.size(0)):
        #     if suppress is not None:
        #         if det_labels[i] in suppress:
        #             continue

        for i in range(len(det_labels)):
            # Boxes
            box_location = det_boxes[i].tolist()

            cv2.rectangle(annotated_image, pt1=(int(box_location[0]), int(box_location[1])),
                          pt2=(int(box_location[2]), int(box_location[3])),
                          color=hex_to_rgb(label_color_map[det_labels[i]]), thickness=2)

            # Text
            text = det_labels[i].upper()
            label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 2)
            text_location = [box_location[0] + 2, box_location[1] + label_size[0][1] + 1]
            textbox_location = [box_location[0], box_location[1] - label_size[0][1],
                                box_location[0] + label_size[0][0] + 4., box_location[1]]
            cv2.rectangle(annotated_image, pt1=(int(textbox_location[0]), int(textbox_location[1])),
                          pt2=(int(textbox_location[0]), int(textbox_location[1])),
                          color=hex_to_rgb(label_color_map[det_labels[i]]), thickness=-1)
            cv2.putText(annotated_image, text, org=(int(text_location[0]), int(text_location[1])),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, thickness=2, fontScale=0.5, color=(255, 255, 255))

        print('wrinting to images:', img_name)
        cv2.imwrite(os.path.join(image_root, img_name).strip(suffix) + '_bbox.jpg', annotated_image)
        del predicted_locs, predicted_scores, image, det_boxes, det_labels, det_scores, original_image


def detect_image(image_path, model, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param image_path:
    :param model:
    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    img_name = image_path
    original_image = Image.open(image_path, mode='r')
    original_image = original_image.convert('RGB')

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    _, predicted_locs, predicted_scores, predicted_points, _ = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores, det_points = model.detect_objects(predicted_locs.clamp_(0, 1), predicted_scores, predicted_points.clamp_(0, 1), min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')
    det_points = det_points[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims.repeat(1, 2)
    det_points = det_points * original_dims.repeat(1, 32)
    # print(det_points, det_boxes)
    # exit()

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.']
    # i.e. ['background'] in SSD300.detect_objects() in model.py
    annotated_image = cv2.imread(image_path)

    if det_labels == ['background']:
        # Just return original image
        print('No object is detected in this image:', img_name)
        return annotated_image

    # Annotate
    for i in range(len(det_labels)):
        # Boxes
        box_location = det_boxes[i].tolist()
        point_location = det_points[i].tolist()
        # print(point_location)

        cv2.rectangle(annotated_image, pt1=(int(box_location[0]), int(box_location[1])),
                      pt2=(int(box_location[2]), int(box_location[3])),
                      color=hex_to_rgb(label_color_map[det_labels[i]]), thickness=2)

        # Text
        text = det_labels[i].upper()
        label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.4, 1)
        text_location = [box_location[0] + 1, box_location[1] + 1, box_location[0] + 1 + label_size[0][0],
                         box_location[1] + 1 + label_size[0][1]]
        cv2.rectangle(annotated_image, pt1=(int(text_location[0]), int(text_location[1])),
                      pt2=(int(text_location[2]), int(text_location[3])),
                      color=(50, 100, 50), thickness=-1)
        cv2.putText(annotated_image, text, org=(int(text_location[0]), int(text_location[3])),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, thickness=1, fontScale=0.4, color=(255, 255, 255))

        for j in range(len(point_location) // 2):

            cv2.circle(annotated_image, center=(int(point_location[2 * j]), int(point_location[2 * j + 1])), radius=3,
                       color=hex_to_rgb(label_color_map[det_labels[i]]), thickness=-1)

    # del predicted_locs, predicted_scores, image, det_boxes, det_labels, det_scores, original_image
    print('wrinting to images:', image_path)
    return annotated_image


def print_help():
    print('Try one of the following options:')
    print('python detect_bbox --folder(detect for all images under the folder)')
    print('python detect_bbox --video(detect for all frames in the video)')
    print('python detect_bbox --image(detect for a single image)')
    print('saved images will be put in the same location as input with some suffix')
    exit()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print_help()
        exit()
    # if sys.argv[1] == '--folder':
    #     img_root = 'data/VOC/Eval_images/detect'
    #     model_path = 'checkpoints/my_checkpoint_deform300_b32.pth.tar'
    #     detect(img_root, model_path, min_score=0.5, max_overlap=0.5, top_k=100)

    if sys.argv[1] == '--image':
        img_root = 'data/VOC/Eval_images/shape'
        # model_path = 'checkpoints/my_checkpoint_anchor_shape_basis64.pth.tar'
        model_path = 'checkpoints/my_checkpoint_refine_shape_basis64.pth.tar'
        output_folder = 'code'
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch'] + 1
        print(model_path)
        print('Loading checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        model = model.to(device)
        model.eval()
        for image_name in os.listdir(img_root):
            if os.path.isdir(os.path.join(img_root, image_name)):
                continue
            annotated_image = detect_image(os.path.join(img_root, image_name), model, min_score=0.25, max_overlap=0.3, top_k=100)
            cv2.imwrite(os.path.join(os.path.join(img_root, output_folder), image_name), annotated_image)
    else:
        raise NotImplementedError