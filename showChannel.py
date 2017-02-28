def show_channels(fname):
    import matplotlib.image as mpimg
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import os.path

    img = mpimg.imread(fname)   # in RGB

    f, ((orig, dummy1, gray), (Rax, Gax, Bax), (Hax, Sax, Lax), (H_ax, S_ax, V_ax)) = plt.subplots(4, 3, figsize=(20, 20))
    f.tight_layout()

    orig.imshow(img)
    orig.set_title("Original: RGB", fontsize=50)

    gray.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cmap='gray')
    gray.set_title("Grayscale", fontsize=50)

    Rax.imshow(img[:,:,0], cmap='gray')
    Rax.set_title("R channel", fontsize=50)

    Gax.imshow(img[:,:,1], cmap='gray')
    Gax.set_title("G channel", fontsize=50)

    Bax.imshow(img[:,:,2], cmap='gray')
    Bax.set_title("B channel", fontsize=50)

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    Hax.imshow(hls[:,:,0], cmap='gray')
    Hax.set_title("H channel in HLS", fontsize=50)

    Sax.imshow(hls[:,:,2], cmap='gray')
    Sax.set_title("S channel in HLS", fontsize=50)

    Lax.imshow(hls[:,:,1], cmap='gray')
    Lax.set_title("L channel in HLS", fontsize=50)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H_ax.imshow(hsv[:,:,0], cmap='gray')
    H_ax.set_title("H channel in HSV", fontsize=50)

    S_ax.imshow(hsv[:,:,1], cmap='gray')
    S_ax.set_title("S channel in HSV", fontsize=50)

    V_ax.imshow(hsv[:,:,2], cmap='gray')
    V_ax.set_title("V channel in HSV", fontsize=50)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.) 
    plt.show()  
    f.savefig("./output_images/channel_" + os.path.basename(fname))
