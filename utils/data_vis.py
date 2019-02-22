import matplotlib.pyplot as plt

import cv2
def plot_img_and_mask(img, mask, gtimg, fn):
    fig = plt.figure()
    a = fig.add_subplot(1, 3, 1)
    a.set_title('Input image')
    plt.imshow(img)
    plt.axis('off')

    b = fig.add_subplot(1, 3, 2)
    b.set_title('Output mask')
    plt.imshow(mask)
    plt.axis('off')

    b = fig.add_subplot(1, 3, 3)
    b.set_title('GT mask')
    plt.imshow(gtimg)
    plt.axis('off')

    plt.savefig('./CombResult/{}.jpg'.format(fn))
    plt.close(fig)
    cv2.imwrite('./OnlyResult/{}.jpg'.format(fn), mask*255)

