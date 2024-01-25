import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms


# 把numpy变为Tensor
def toTensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256


# 把Tensor变为numpy
def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img


# 显示mask值区域
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


# 显示标定点
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


image = cv2.imread('groceries.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8.5, 6))
plt.imshow(image)
plt.title('origin_img')
plt.axis('off')
plt.show()

import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# 直接篡改
predictor = SamPredictor(sam)
predictor.set_image(image)
input_point = np.array([[500, 300]])
input_label = np.array([1])
plt.figure(figsize=(8.5, 6))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

for i, (mask, score) in enumerate(zip(masks, scores)):
    if i == 2:
        plt.figure(figsize=(8.5, 6))
        img = toTensor(image)
        mask_tensor = torch.BoolTensor(mask)
        img_mask = torch.masked_select(img, mask_tensor)
        im = toTensor(cv2.cvtColor(cv2.imread('dog.jpg'), cv2.COLOR_BGR2RGB))
        im = torch.masked_scatter(im, mask_tensor, img_mask)
        plt.imshow(tensor_to_np(im))
        plt.axis('off')
        plt.show()

# 平移后篡改
predictor = SamPredictor(sam)
predictor.set_image(image)
input_point = np.array([[500, 267 - (300 - 267)]])
input_label = np.array([1])
plt.figure(figsize=(8.5, 6))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

for i, (mask, score) in enumerate(zip(masks, scores)):
    if i == 2:
        plt.figure(figsize=(8.5, 6))
        img = toTensor(image)
        mask_tensor = torch.BoolTensor(mask)
        img_mask = torch.masked_select(img, mask_tensor)
        im = toTensor(cv2.cvtColor(cv2.imread('dog.jpg'), cv2.COLOR_BGR2RGB))
        masks_tensors = torch.chunk(mask_tensor, 4, 1)
        mask_tensor = torch.cat([masks_tensors[0], masks_tensors[2], masks_tensors[1], masks_tensors[3]], 1)
        im = torch.masked_scatter(im, mask_tensor, img_mask)
        plt.imshow(tensor_to_np(im))
        plt.axis('off')
        plt.show()

# 横向翻转
transform_horizontalFlip = transforms.RandomHorizontalFlip(1)
image_horizontalFlip = tensor_to_np(transform_horizontalFlip(toTensor(image)))

predictor = SamPredictor(sam)
predictor.set_image(image_horizontalFlip)
input_point = np.array([[400 - (500 - 400), 300]])
input_label = np.array([1])
plt.figure(figsize=(8.5, 6))
plt.imshow(image_horizontalFlip)
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

for i, (mask, score) in enumerate(zip(masks, scores)):
    if i == 2:
        plt.figure(figsize=(8.5, 6))
        img = toTensor(image_horizontalFlip)
        mask_tensor = torch.BoolTensor(mask)
        img_mask = torch.masked_select(img, mask_tensor)
        im = toTensor(cv2.cvtColor(cv2.imread('dog.jpg'), cv2.COLOR_BGR2RGB))
        im = torch.masked_scatter(im, mask_tensor, img_mask)
        plt.imshow(tensor_to_np(im))
        plt.axis('off')
        plt.show()

# 纵向翻转
transform_verticalFlip = transforms.RandomVerticalFlip(1)
image_verticalFlip = tensor_to_np(transform_verticalFlip(toTensor(image)))

predictor = SamPredictor(sam)
predictor.set_image(image_verticalFlip)
input_point = np.array([[500, 267 - (300 - 267)]])
input_label = np.array([1])
plt.figure(figsize=(8.5, 6))
plt.imshow(image_verticalFlip)
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

for i, (mask, score) in enumerate(zip(masks, scores)):
    if i == 2:
        plt.figure(figsize=(8.5, 6))
        img = toTensor(image_verticalFlip)
        mask_tensor = torch.BoolTensor(mask)
        img_mask = torch.masked_select(img, mask_tensor)
        im = toTensor(cv2.cvtColor(cv2.imread('dog.jpg'), cv2.COLOR_BGR2RGB))
        im = torch.masked_scatter(im, mask_tensor, img_mask)
        plt.imshow(tensor_to_np(im))
        plt.axis('off')
        plt.show()

# 逆时针旋转45度
ang = 45
transform_anticlockwise45rotation = transforms.RandomRotation(degrees=(ang, ang), center=(400, 267))
image_anticlockwise45rotation = tensor_to_np(transform_anticlockwise45rotation(toTensor(image)))

predictor = SamPredictor(sam)
predictor.set_image(image_anticlockwise45rotation)
angle = -ang * (math.pi / 180)
input_point = np.array([[int((500 - 400) * math.cos(angle) - (300 - 267) * math.sin(angle) + 400),
                         int((500 - 400) * math.sin(angle) + (300 - 267) * math.cos(angle) + 267)]])
input_label = np.array([1])
plt.figure(figsize=(8.5, 6))
plt.imshow(image_anticlockwise45rotation)
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

for i, (mask, score) in enumerate(zip(masks, scores)):
    if i == 2:
        plt.figure(figsize=(8.5, 6))
        img = toTensor(image_anticlockwise45rotation)
        mask_tensor = torch.BoolTensor(mask)
        img_mask = torch.masked_select(img, mask_tensor)
        im = toTensor(cv2.cvtColor(cv2.imread('dog.jpg'), cv2.COLOR_BGR2RGB))
        im = torch.masked_scatter(im, mask_tensor, img_mask)
        plt.imshow(tensor_to_np(im))
        plt.axis('off')
        plt.show()

# 顺时针旋转45度
ang = -45
transform_clockwise45rotation = transforms.RandomRotation(degrees=(ang, ang), center=(400, 267))
image_clockwise45rotation = tensor_to_np(transform_clockwise45rotation(toTensor(image)))

predictor = SamPredictor(sam)
predictor.set_image(image_clockwise45rotation)
angle = -ang * (math.pi / 180)
input_point = np.array([[int((500 - 400) * math.cos(angle) - (300 - 267) * math.sin(angle) + 400),
                         int((500 - 400) * math.sin(angle) + (300 - 267) * math.cos(angle) + 267)]])
input_label = np.array([1])
plt.figure(figsize=(8.5, 6))
plt.imshow(image_clockwise45rotation)
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

for i, (mask, score) in enumerate(zip(masks, scores)):
    if i == 2:
        plt.figure(figsize=(8.5, 6))
        img = toTensor(image_clockwise45rotation)
        mask_tensor = torch.BoolTensor(mask)
        img_mask = torch.masked_select(img, mask_tensor)
        im = toTensor(cv2.cvtColor(cv2.imread('dog.jpg'), cv2.COLOR_BGR2RGB))
        im = torch.masked_scatter(im, mask_tensor, img_mask)
        plt.imshow(tensor_to_np(im))
        plt.axis('off')
        plt.show()

# 放大
shape = image.shape
transform_blowup = transforms.Compose([
    transforms.Resize((int(shape[0] * 2), int(shape[1] * 2))),
    transforms.CenterCrop((int(shape[0]), int(shape[1])))]
)
image_blowup = tensor_to_np(transform_blowup(toTensor(image)))

predictor = SamPredictor(sam)
predictor.set_image(image_blowup)
input_point = np.array([[int((500 - 400) * 2) + 400, int((300 - 267) * 2) + 267]])
input_label = np.array([1])
plt.figure(figsize=(8.5, 6))
plt.imshow(image_blowup)
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

for i, (mask, score) in enumerate(zip(masks, scores)):
    if i == 2:
        plt.figure(figsize=(8.5, 6))
        img = toTensor(image_blowup)
        mask_tensor = torch.BoolTensor(mask)
        img_mask = torch.masked_select(img, mask_tensor)
        im = toTensor(cv2.cvtColor(cv2.imread('dog.jpg'), cv2.COLOR_BGR2RGB))
        im = torch.masked_scatter(im, mask_tensor, img_mask)
        plt.imshow(tensor_to_np(im))
        plt.axis('off')
        plt.show()

# 缩小
shape = image.shape
transform_shrink = transforms.Compose([
    transforms.Resize((int(shape[0] / 2), int(shape[1] / 2))),
    transforms.Pad((int(shape[1] / 4), int(shape[0] / 4),
                    shape[1] - int(shape[1] / 2) - int(shape[1] / 4),
                    shape[0] - int(shape[0] / 2) - int(shape[0] / 4)))]
)
image_shrink = tensor_to_np(transform_shrink(toTensor(image)))

predictor = SamPredictor(sam)
predictor.set_image(image_shrink)
input_point = np.array([[(500 - 400) / 2 + 400, int((300 - 267) / 2) + 267]])
input_label = np.array([1])
plt.figure(figsize=(8.5, 6))
plt.imshow(image_shrink)
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

for i, (mask, score) in enumerate(zip(masks, scores)):
    if i == 2:
        plt.figure(figsize=(8.5, 6))
        img = toTensor(image_shrink)
        mask_tensor = torch.BoolTensor(mask)
        img_mask = torch.masked_select(img, mask_tensor)
        im = toTensor(cv2.cvtColor(cv2.imread('dog.jpg'), cv2.COLOR_BGR2RGB))
        im = torch.masked_scatter(im, mask_tensor, img_mask)
        plt.imshow(tensor_to_np(im))
        plt.axis('off')
        plt.show()
