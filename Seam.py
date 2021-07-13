import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob



def sobel(img):
    img = img.astype(np.float32)
    dx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    dy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    e = (np.sqrt(dx**2 + dy**2)/np.sqrt(2)).astype(np.uint8)
    sobel_img= cv2.applyColorMap(e,cv2.COLORMAP_HSV)
    cv2.imwrite('result_sobel.jpg', sobel_img)
    return e

def energy_ObjectRemoval(img_gray):
    e = sobel(img_gray)
    n,m = e.shape
    #coord_list = [[112, 12], [198, 397]]
    #coord_list_1 = [[0, 15], [226, 47]] #for 1.jpg
    #coord_list_3 = [[10, 45], [200, 84]] #for 3.jpg
    #e[:,21:28] = 0 #for 1.jpg
    e[:,45:82] = 0 #for 3.jpg
    #print(e.shape)
    return e

def vertical_seam(img1, object_removal_flag):
    # Calculate CME
    n,m = img1.shape
    #print(img1.shape)
    if object_removal_flag:
        e = energy_ObjectRemoval(img1)
    else:
        e = sobel(img1)
    M = np.zeros((n,m),dtype='uint32')
    M[0,:] = e[0,:]
    
    for i in range(1,n):
        for j in range(0,m):
            if j==0:
                 M[i,j] = e[i,j] + min(M[i-1,j],M[i-1,j+1])
            elif j==m-1:
                 M[i,j] = e[i,j] + min(M[i-1,j-1],M[i-1,j])
            else:
                 M[i,j] = e[i,j] + min(M[i-1,j-1],M[i-1,j],M[i-1,j+1])


    # Find optimal seam
    seam = np.zeros((n), dtype = 'uint32')
    #print(np.argmin(M[n-1,:]))
    seam[n-1] = np.argmin(M[n-1,:])
    for i in range(n-1,0,-1):
        
            if seam[i]==0:
                seam[i-1] = np.argmin(M[i-1,seam[i]:seam[i]+2]) + seam[i]
            elif seam[i]==m-1:
                seam[i-1] = np.argmin(M[i-1,seam[i]-1:seam[i]+1]) + seam[i]-1
            else:
                seam[i-1] = np.argmin(M[i-1,seam[i]-1:seam[i]+2]) + seam[i]-1
    
    
    return seam

def horizontal_seam(img1):
    # Calculate CME
    n,m = img1.shape
    #print(img1.shape)
    #e = energy(img1)
    e = sobel(img1)
    #e = energy_ObjectRemoval(img1)
    M = np.zeros((n,m),dtype='uint32')
    M[:,0] = e[:,0]
    
    for i in range(1,m):
        for j in range(0,n):
            if j==0:
                 M[j,i] = e[j,i] + min(M[j,i-1],M[j+1,i-1])
            elif j==n-1:
                 M[j,i] = e[j,i] + min(M[j,i-1],M[j-1,i-1])
            else:
                 M[j,i] = e[j,i] + min(M[j,i-1],M[j-1,i-1],M[j+1,i-1])


    # Find optimal seam
    seam = np.zeros((m), dtype = 'uint32')
    #print(np.argmin(M[n-1,:]))
    seam[m-1] = np.argmin(M[:,m-1])
    for i in range(m-1,0,-1):
        
            if seam[i]==0:
                seam[i-1] = np.argmin(M[seam[i]:seam[i]+2,i-1]) + seam[i]
            elif seam[i]==n-1:
                seam[i-1] = np.argmin(M[seam[i]-1:seam[i]+1,i-1]) + seam[i]-1
            else:
                seam[i-1] = np.argmin(M[seam[i]-1:seam[i]+2,i-1]) + seam[i]-1
        
    return seam

def carve_seam_hori(img_gray, img_rgb, seam, counter, img):
    n,m = img_gray.shape
    
    n1, m1,_ = img_rgb.shape
    new_img_rgb = np.zeros((n1-1,m1,3), dtype ='uint8')
    new_img_gray = np.zeros((n1-1,m1), dtype ='uint8')
    progress_img = np.zeros(img.shape, dtype ='uint8')
    progress_img[:n1, :m1] = img_rgb
    for j in range(m1):
        index = seam[j]
        progress_img[index, j] = (255,0,0)
        new_img_rgb[:index,j] = img_rgb[:index,j]
        new_img_rgb[index:,j] = img_rgb[index+1:,j]
        
        new_img_gray[:index,j] = img_gray[:index,j]
        new_img_gray[index:,j] = img_gray[index+1:,j]
    
    if counter != -1:
        write1 = cv2.cvtColor(progress_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('progress'+str(counter)+'.png', write1) 
        
        progress_img[:n1-1,:] = new_img_rgb[:,:]
        counter += 1
        write1 = cv2.cvtColor(progress_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('progress'+str(counter)+'.png', write1) 
    return new_img_gray,new_img_rgb

def carve_seam_vert(img_gray, img_rgb, seam):
    n,m = img_gray.shape
    mask = np.ones((n, m), dtype=np.bool)
    for i in range(n):
        mask[i, seam[i]] = False
    
    img_gray1 = img_gray[mask].reshape((n, m - 1))
    img_rgb = img_rgb[mask].reshape((n,m-1,3))
    return img_gray1,img_rgb

def reduce_width(img_gray, img_rgb,s, img):
    n,m = img_gray.shape
    #print(n,m)
    img_gray1 = img_gray.copy()
     
    seam_list = []
    for i in range(s):
        seam = vertical_seam(img_gray1,0)
        seam_list.append(seam)
        img_gray1,img_rgb = carve_seam_vert(img_gray1, img_rgb, seam)
        #print(i)
        
    #draw seam
    for i in range(len(seam_list)):
        seam1 = seam_list[i]
        if i > 0:
            seam1[np.where(seam1 > seam_list[i-1])] += 1
        for i in range(img.shape[0]):
            img[i,seam1[i]] = (255,0,0)
    result_seam = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('result_seams_vertical.jpg', result_seam) 

    plt.imshow(img_rgb)
    save_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite('result.jpg', save_img)
    return img_rgb

def reduce_height(img_gray, img_rgb,s,img):
    n,m = img_gray.shape
    img_gray1 = img_gray.copy()
  
    seam_list = []
    counter = 0
    for i in range(s):
        seam = horizontal_seam(img_gray1)
        seam_list.append(seam)
        img_gray1,img_rgb = carve_seam_hori(img_gray1, img_rgb, seam, counter, img.copy())
        counter += 2
        #print(i)
    
    #draw seam
    for i in range(len(seam_list)):
        seam1 = seam_list[i]
        if i > 0:
            seam1[np.where(seam1 >= seam_list[i-1])] += 1
        for i in range(img.shape[1]):
            img[seam1[i],i] = (255,0,0)
    result_seam = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('result_seams_horizontal.jpg', result_seam)  

    plt.imshow(img_rgb)
    save_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite('result.jpg', save_img)
    
    
    #gif
    import os
    import imageio
    file_name = []
    images = []
    for filename in glob.glob("progress*"):
        file_name.append(filename)
    file_name.sort(key=lambda x : int(x[8:-4]))
    for filename in file_name:
        images.append(imageio.imread(filename))
    imageio.mimsave('Gif_progress.gif', images, fps = 5)
    
    #remove images
    for filename in glob.glob("progress*"):
        os.remove(filename) 
            
    return img_rgb
    
def increase_width(img_gray, img_rgb,s,img):         
        
    n,m = img_gray.shape
    img_gray1 = img_gray.copy()
    img_rgb1 = img_rgb.copy()
   
    seam_list = []
    # get s seams
    for i in range(s):
        seam = vertical_seam(img_gray1,0)
        seam_list.append(seam)
        img_gray1,img_rgb1 = carve_seam_vert(img_gray1, img_rgb1, seam)
   
    seam_list.reverse()
     #draw seam
    for i in range(len(seam_list)):
        seam1 = seam_list[i]
        for i in range(img.shape[0]):
            img[i,seam1[i]] = (255,0,0)
    result_seam = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('result_seams_vertical.jpg', result_seam) 
      
    for i in range(len(seam_list)):
        
        seam = seam_list[i]
        if i > 0:
            seam[np.where(seam >= seam_list[i-1])] += 1       
        
        n1, m1,_ = img_rgb.shape
        new_img = np.zeros((n1,m1+1,3), dtype ='uint8')
        for j in range(n1):
            index = seam[j]
            new_img[j,:index+1] = img_rgb[j,:index+1]
            new_img[j,index+1] = (img_rgb[j,index])
            new_img[j,index+2:] = img_rgb[j,index+1:]
        
        img_rgb = new_img.copy()
        
    save_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite('result.jpg', save_img)
    return img_rgb

def increase_height(img_gray, img_rgb,s,img):         
        
    n,m = img_gray.shape
    img_gray1 = img_gray.copy()
    img_rgb1 = img_rgb.copy()

    seam_list = []
    # get s seams
    counter = 0
    for i in range(s):
        seam = horizontal_seam(img_gray1)
        seam_list.append(seam)
        img_gray1,img_rgb1 = carve_seam_hori(img_gray1, img_rgb1, seam, -1, img.copy())
        counter += 2
    
    seam_list.reverse()
     #draw seam
    for i in range(len(seam_list)):
        seam1 = seam_list[i]
    
        for i in range(img.shape[1]):
            img[seam1[i],i] = (255,0,0)
    result_seam = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('result_seams_horizontal.jpg', result_seam)         
    for i in range(len(seam_list)):
        
        seam = seam_list[i]
            
        n1, m1,_ = img_rgb.shape
        new_img = np.zeros((n1+1,m1,3), dtype ='uint8')
        for j in range(m1):
            index = seam[j]
            new_img[:index+1,j] = img_rgb[:index+1,j]
            new_img[index+1,j] = (img_rgb[index,j])
            new_img[index+2:,j] = img_rgb[index+1:,j]
        
        img_rgb = new_img.copy()
    
    plt.imshow(img_rgb)
    save_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite('result.jpg', save_img)
        
    return img_rgb

def object_removal(img_gray,img_rgb):
    n,m = img_gray.shape
    img_gray1 = img_gray.copy()
    #s = 7  # 1.jpg
    s = 37 # 3.jpg
    #print(s)
    seam_list = []
    
    for i in range(s):
        seam = vertical_seam(img_gray1,1)
        seam_list.append(seam)
        img_gray1,img_rgb = carve_seam_vert(img_gray1, img_rgb, seam)
        #print(i)
             

    plt.imshow(img_rgb)
    save_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite('result_ObjectRemoval.jpg', save_img)
    return img_rgb
    
if __name__ == '__main__':
 
    img = cv2.imread('result_ObjectRemoval.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img1 = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_rgb = img.copy()
    
    f, axarr = plt.subplots(1,2) 
    axarr[0].imshow(img_rgb)
    
    case = int(input("enter 1 for resize or 2 for object removal\n"))
    if case == 1:
        #size
        n,m,_ = img.shape
        print("current image size {} {}".format(img.shape[0],img.shape[1]))
        n1,m1 = map(int, input("enter new image size\n").split())
        
        if n1 > n:
            img_rgb = increase_height(img_gray,img_rgb, n1-n,img)
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            
        elif n1 < n:
            img_rgb = reduce_height(img_gray, img_rgb, n-n1,img)
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            
        if  m1 < m:
            img_rgb = reduce_width(img_gray, img_rgb, m-m1,img_rgb.copy())
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            
        elif m1 > m:
            img_rgb = increase_width(img_gray,img_rgb, m1-m,img_rgb.copy())
        
        
       
      
    #remove object          
    else:
        img_rgb = object_removal(img_gray, img_rgb)
    
    axarr[1].imshow(img_rgb)