def display_side_by_side(orig, processed, cmap=None, caption="processed", save_path=None):
      # assume save_path does not have '/' at the end

      import matplotlib.pyplot as plt

      f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9)) # 1 row, 2 columns, with alias ax1, ax2
      f.tight_layout()

      ax1.imshow(orig) # show image at ax1
      ax1.set_title("Original image", fontsize=50)

      ax2.imshow(processed, cmap=cmap) 
      ax2.set_title(caption, fontsize=50)

      plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.) 
      plt.show()                  # This is needed to make the plot visible when executing from shell

      if save_path:
          f.savefig(save_path+'/'+caption+'.jpg', bbox_inches='tight', pad_inches=0.0, dpi=200,)

def normalized(img):
    import numpy as np
    return np.uint8(255*img/np.max(np.absolute(img)))
