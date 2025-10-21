from DeepVOG_model import load_DeepVOG
import skimage.io as ski
import numpy as np
from skimage.transform import resize


def test_if_model_work():
    model = load_DeepVOG()
    img_data = ski.imread("S1001L01.jpg") / 255.0
    img_resized = resize(img_data, (240, 320), anti_aliasing=True)
    img = np.zeros((1, 240, 320, 3))
    img[:,:,:,:] = (img_resized).reshape(1, 240, 320, 1)
    prediction = model.predict(img)
    pred = prediction[0, :, :, 1]
    pred_uint8 = (pred * 255)
    pred_uint8 = resize(pred_uint8, (img_data.shape[0], img_data.shape[1]), anti_aliasing=True)
    pred_uint8 = (pred_uint8).astype(np.uint8)
    threshold = 127
    pred_uint8_binary = np.zeros(pred_uint8.shape, dtype=np.uint8)
    pred_uint8_binary[pred_uint8 <= threshold] = 0
    pred_uint8_binary[pred_uint8 > threshold] = 255
    ski.imsave("test_prediction.png", pred_uint8)
    ski.imsave("test_prediction_binary.png", pred_uint8_binary)

if __name__ == "__main__":
    # If model works, the "test_prediction.png" should show the segmented area of pupil from "test_image.png"
    test_if_model_work()

