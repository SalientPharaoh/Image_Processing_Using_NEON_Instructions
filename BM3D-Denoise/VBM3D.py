import cv2
import numpy
import multiprocessing

cv2.setUseOptimized(True)

# Parameters initialization
Frame_number=10
reference_frame_num=7
sigma = 25
Threshold_Hard3D = 2.7*sigma           
First_Match_threshold = 5000            
Step1_max_matched_cnt = 16 
Step1_Blk_Size = 8 
Step1_Blk_Step = 3 
Step1_Search_Step = 3 
Step1_Search_Window = 39  

Second_Match_threshold = 5000  
Step2_max_matched_cnt = 16
Step2_Blk_Size = 8
Step2_Blk_Step = 3
Step2_Search_Step = 3
Step2_Search_Window = 39

Beta_Kaiser = 2.0


def init(img, _blk_size, _Beta_Kaiser):
    m_shape = img.shape
    m_img = numpy.zeros([m_shape[0],m_shape[1],Frame_number], dtype=float)
    m_wight = numpy.zeros([m_shape[0],m_shape[1],Frame_number], dtype=float)
    K = numpy.matrix(numpy.kaiser(_blk_size, _Beta_Kaiser))
    m_Kaiser = numpy.array(K.T * K)          
    return m_img, m_wight, m_Kaiser
def init_step2(img, _blk_size, _Beta_Kaiser):
    m_shape = img.shape
    m_img = numpy.zeros([m_shape[0],m_shape[1],Frame_number], dtype=float)
    m_wight = numpy.zeros([m_shape[0],m_shape[1],Frame_number], dtype=float)
    K = numpy.matrix(numpy.kaiser(_blk_size, _Beta_Kaiser))
    m_Kaiser = numpy.array(K.T * K)         
    return m_img, m_wight, m_Kaiser

def Locate_blk(i, j, blk_step, block_Size, width, height, frame):
    if i*blk_step+block_Size < width:
        point_x = i*blk_step
    else:
        point_x = width - block_Size

    if j*blk_step+block_Size < height:
        point_y = j*blk_step
    else:
        point_y = height - block_Size

    m_blockPoint = numpy.array((point_x, point_y, frame), dtype=int)

    return m_blockPoint


def Define_SearchWindow(_noisyImg, _BlockPoint, _WindowSize, Blk_Size):
    point_x = _BlockPoint[0] 
    point_y = _BlockPoint[1] 

    LX = point_x+Blk_Size/2-_WindowSize/2    
    LY = point_y+Blk_Size/2-_WindowSize/2    
    RX = LX+_WindowSize                   
    RY = LY+_WindowSize  

    if LX < 0:   LX = 0
    elif RX >= _noisyImg.shape[0]:   LX = _noisyImg.shape[0]-_WindowSize
    if LY < 0:   LY = 0
    elif RY >= _noisyImg.shape[0]:   LY = _noisyImg.shape[0]-_WindowSize

    return numpy.array((LX, LY), dtype=int)


def Step1_fast_match(frames, _BlockPoint):
    (present_x, present_y ,frame) = _BlockPoint 
    Blk_Size = Step1_Blk_Size
    Search_Step = Step1_Search_Step
    Threshold = First_Match_threshold
    max_matched = Step1_max_matched_cnt
    Window_size = Step1_Search_Window

    blk_positions = numpy.zeros((max_matched, 3), dtype=int) 
    Final_similar_blocks = numpy.zeros((max_matched, Blk_Size, Blk_Size), dtype=float)

    _noisyImg=frames[frame]
    _noisyImg=_noisyImg[:,:,0]
    img = _noisyImg[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
    dct_img = cv2.dct(img.astype(numpy.float32)) 


    Final_similar_blocks[0, :, :] = dct_img
    blk_positions[0, :] = (present_x, present_y, frame)

    Window_location = Define_SearchWindow(_noisyImg, _BlockPoint, Window_size, Blk_Size)
    blk_num = ((Window_size*15-Blk_Size)/Search_Step)*((Window_size-Blk_Size)/Search_Step)
    blk_num = int(blk_num)
    (present_x, present_y) = Window_location

    similar_blocks = numpy.zeros((blk_num, Blk_Size, Blk_Size), dtype=float)
    m_Blkpositions = numpy.zeros((blk_num, 3), dtype=int)
    Distances = numpy.zeros(blk_num, dtype=float) 
    blk_num_serch=int((Window_size-Blk_Size)/Search_Step)
    matched_cnt = 0
    if frame<numpy.floor(reference_frame_num/2):
        start_frame=0
    elif frame+numpy.floor(reference_frame_num/2)>Frame_number:
        start_frame=numpy.uint8(Frame_number-numpy.floor(reference_frame_num/2))
    else :
        start_frame=numpy.uint8(frame-numpy.floor(reference_frame_num/2))
    for m_frame in range(start_frame,min(numpy.uint8(start_frame+numpy.floor(reference_frame_num)),Frame_number)):
        _noisyImg_frame=frames[m_frame]
        _noisyImg_frame=_noisyImg_frame[:,:,0]
        for i in range(blk_num_serch):
            for j in range(blk_num_serch):
                Tem_img = _noisyImg_frame[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
                dct_Tem_img = cv2.dct(Tem_img.astype(numpy.float32)) 
                m_Distance = numpy.linalg.norm((dct_img-dct_Tem_img))**2 / (Blk_Size**2)
                if m_Distance < Threshold and m_Distance > 0:
                    similar_blocks[matched_cnt, :, :] = dct_Tem_img
                    m_Blkpositions[matched_cnt, :] = (present_x, present_y, m_frame)
                    Distances[matched_cnt] = m_Distance
                    matched_cnt += 1
                present_y += Search_Step
            present_x += Search_Step
            present_y = Window_location[1]
        present_x = Window_location[0]
        present_y = Window_location[1]
    Distances = Distances[:matched_cnt]
    Sort = Distances.argsort()

    if matched_cnt < max_matched:
        Count = matched_cnt + 1
    else:
        Count = max_matched

    if Count > 0:
        for i in range(1, Count):
            Final_similar_blocks[i, :, :] = similar_blocks[Sort[i-1], :, :]
            blk_positions[i, :] = m_Blkpositions[Sort[i-1], :]
    return Final_similar_blocks, blk_positions, Count


def Step1_3DFiltering(_similar_blocks):
    statis_nonzero = 0  
    m_Shape = _similar_blocks.shape
    for i in range(m_Shape[1]):
        for j in range(m_Shape[2]):
            tem_Vct_Trans = cv2.dct(_similar_blocks[:, i, j])
            tem_Vct_Trans[numpy.abs(tem_Vct_Trans[:]) < Threshold_Hard3D] = 0.
            statis_nonzero += tem_Vct_Trans.nonzero()[0].size
            _similar_blocks[:, i, j] = cv2.idct(tem_Vct_Trans)[:,0]
    return _similar_blocks, statis_nonzero


def Aggregation_hardthreshold(_similar_blocks, blk_positions, m_basic_img, m_wight_img, _nonzero_num, Count, Kaiser):
    _shape = _similar_blocks.shape
    if _nonzero_num < 1:
        _nonzero_num = 1
    block_wight = (1./_nonzero_num) * Kaiser
    for i in range(Count):
        point = blk_positions[i, :]
        tem_img = (1./_nonzero_num)*cv2.idct(_similar_blocks[i, :, :]) * Kaiser
        m_basic_img[point[0]:point[0]+_shape[1], point[1]:point[1]+_shape[2], point[2]] += tem_img
        m_wight_img[point[0]:point[0]+_shape[1], point[1]:point[1]+_shape[2], point[2]] += block_wight

def BM3D_1st_step(frames):
    (width, height,channel) = frames[0].shape   
    block_Size = Step1_Blk_Size    
    blk_step = Step1_Blk_Step       
    Width_num = (width - block_Size)/blk_step
    Height_num = (height - block_Size)/blk_step
    Basic_img, m_Wight, m_Kaiser = init(frames[0][:,:,0], Step1_Blk_Size, Beta_Kaiser)
    Basic_img_tmp = Basic_img
    m_Wight_tmp = m_Wight
    for frame in range(Frame_number):
        for i in range(int(Width_num + 2)):
            for j in range(int(Height_num + 2)):
                m_blockPoint = Locate_blk(i, j, blk_step, block_Size, width, height, frame)
                Similar_Blks, Positions, Count = Step1_fast_match(frames, m_blockPoint)
                Similar_Blks, statis_nonzero = Step1_3DFiltering(Similar_Blks)
                Aggregation_hardthreshold(Similar_Blks, Positions, Basic_img_tmp, m_Wight_tmp, statis_nonzero, Count,
                                          m_Kaiser)
    return Basic_img_tmp/m_Wight_tmp


def Step2_fast_match(Basic_img, frames, _BlockPoint):
    (present_x, present_y, frame) = _BlockPoint 
    Blk_Size = Step2_Blk_Size
    Threshold = Second_Match_threshold
    Search_Step = Step2_Search_Step
    max_matched = Step2_max_matched_cnt
    Window_size = Step2_Search_Window

    blk_positions = numpy.zeros((max_matched, 3), dtype=int) 
    Final_similar_blocks = numpy.zeros((max_matched, Blk_Size, Blk_Size), dtype=float)
    Final_noisy_blocks = numpy.zeros((max_matched, Blk_Size, Blk_Size), dtype=float)

    _Basic_img=Basic_img[:,:,frame]
    img = _Basic_img[present_x: present_x+Blk_Size, present_y: present_y+Blk_Size]
    dct_img = cv2.dct(numpy.float32(img))  
    Final_similar_blocks[0, :, :] = dct_img
    _noisyImg=frames[frame]
    n_img = _noisyImg[present_x: present_x+Blk_Size, present_y: present_y+Blk_Size,0]
    dct_n_img = cv2.dct(n_img.astype(numpy.float32)) 
    Final_noisy_blocks[0, :, :] = dct_n_img

    blk_positions[0, :] =_BlockPoint

    Window_location = Define_SearchWindow(_noisyImg, _BlockPoint, Window_size, Blk_Size)
    blk_num = (Window_size-Blk_Size)/Search_Step 
    blk_num = int(blk_num)
    (present_x, present_y) = Window_location

    similar_blocks = numpy.zeros((blk_num**3, Blk_Size, Blk_Size), dtype=float)
    m_Blkpositions = numpy.zeros((blk_num**3, 3), dtype=int)
    Distances = numpy.zeros(blk_num**3, dtype=float) 

    matched_cnt = 0
    if frame < numpy.floor(reference_frame_num / 2):
        start_frame = 0
    elif frame + numpy.floor(reference_frame_num / 2) > Frame_number:
        start_frame = numpy.uint8(Frame_number - numpy.floor(reference_frame_num / 2))
    else:
        start_frame = numpy.uint8(frame - numpy.floor(reference_frame_num / 2))
    for m_frame in range(start_frame,
                         min(numpy.uint8(start_frame + numpy.floor(reference_frame_num))+1, Frame_number)):
        _noisyImg_frame = Basic_img[:,:,m_frame]
        for i in range(blk_num):
            for j in range(blk_num):
                Tem_img = _noisyImg_frame[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
                dct_Tem_img = cv2.dct(Tem_img.astype(numpy.float32))
                m_Distance = numpy.linalg.norm((dct_img - dct_Tem_img)) ** 2 / (Blk_Size ** 2)
                if m_Distance < Threshold and m_Distance > 0:
                    similar_blocks[matched_cnt, :, :] = dct_Tem_img
                    m_Blkpositions[matched_cnt, :] = (present_x, present_y, m_frame)
                    Distances[matched_cnt] = m_Distance
                    matched_cnt += 1
                present_y += Search_Step
            present_x += Search_Step
            present_y = Window_location[1]
        present_x = Window_location[0]
        present_y = Window_location[1]
    Distances = Distances[:matched_cnt]
    Sort = Distances.argsort()

    if matched_cnt < max_matched:
        Count = matched_cnt + 1
    else:
        Count = max_matched

    if Count > 0:
        for i in range(1, Count):
            Final_similar_blocks[i, :, :] = similar_blocks[Sort[i-1], :, :]
            blk_positions[i, :] = m_Blkpositions[Sort[i-1], :]

            (present_x, present_y, m_frame) = m_Blkpositions[Sort[i-1], :]
            n_img = frames[m_frame][present_x: present_x+Blk_Size, present_y: present_y+Blk_Size,0]
            Final_noisy_blocks[i, :, :] = cv2.dct(n_img.astype(numpy.float64))

    return Final_similar_blocks, Final_noisy_blocks, blk_positions, Count


def Step2_3DFiltering(_Similar_Bscs, _Similar_Imgs):
    m_Shape = _Similar_Bscs.shape
    d31 = numpy.zeros([m_Shape[0], m_Shape[1]])
    d32 = numpy.zeros([m_Shape[0], m_Shape[1]])
    for i in range(m_Shape[1]):
        d31=_Similar_Imgs[:,i,:]
        d32 = _Similar_Bscs[:, i, :]
        d31=cv2.dct(d31)
        d32 = cv2.dct(d32)
        d31 = ((d32**2)/(d32**2+(sigma+15)**2))*d31
        _Similar_Imgs[:, i, :]=d31
        _Similar_Bscs[:, i, :]=d32
    return _Similar_Imgs, _Similar_Bscs


def Aggregation_Wiener(Similar_Imgs, _Wiener_wight, blk_positions, m_basic_img, m_wight_img, Count, Kaiser):
    _shape = Similar_Imgs.shape
    block_wight = _Wiener_wight * Kaiser

    for i in range(Count):
        point = blk_positions[i, :]
        tem_img = _Wiener_wight * cv2.idct(Similar_Imgs[i, :, :]) * Kaiser
        m_basic_img[point[0]:point[0]+_shape[1], point[1]:point[1]+_shape[2], point[2]] += tem_img
        m_wight_img[point[0]:point[0]+_shape[1], point[1]:point[1]+_shape[2], point[2]] += block_wight

def calculate_weight(_Similar_Imgs, group):
    m_Shape = _Similar_Imgs.shape
    group = group.flatten()
    sum_of_squares =group ** 2
    norm_value = numpy.linalg.norm(sum_of_squares / (sum_of_squares + (sigma+15) ** 2)) ** 2
    weight = 1 / (norm_value * (sigma+15) ** 2)
    for i in range(m_Shape[1]):
        d31=_Similar_Imgs[:, i, :]
        d31 = cv2.idct(d31)
        _Similar_Imgs[:, i, :]=d31
    return _Similar_Imgs, weight

def BM3D_2nd_step(Basic_img,frames):
    (width, height,channel) = frames[0].shape
    block_Size = Step2_Blk_Size
    blk_step = Step2_Blk_Step
    Width_num = (width - block_Size)/blk_step
    Height_num = (height - block_Size)/blk_step

    m_img, m_Wight, m_Kaiser = init_step2(frames[0], block_Size, Beta_Kaiser)
    for frame in range(0, Frame_number):
        for i in range(int(Width_num+2)):
            for j in range(int(Height_num+2)):
                m_blockPoint = Locate_blk(i, j, blk_step, block_Size, width, height,frame)
                Similar_Blks, Similar_Imgs, Positions, Count = Step2_fast_match(Basic_img, frames, m_blockPoint)
                _Similar_Imgs, _Similar_Bscs = Step2_3DFiltering(Similar_Blks, Similar_Imgs)
                _Similar_Imgs, weight = calculate_weight(_Similar_Imgs, _Similar_Bscs[:])
                Aggregation_Wiener(_Similar_Imgs, weight, Positions, m_img, m_Wight, Count, m_Kaiser)
    return m_img/m_Wight



if __name__ == '__main__':
    cv2.setUseOptimized(True)  
    filename=""
    video_name =filename + ".mp4"
    cap = cv2.VideoCapture(video_name)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        exit()
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:       
            break
        frames.append(frame)      
    cap.release() 
    e1 = cv2.getTickCount() 
    Basic_img = BM3D_1st_step(frames)
    for frame in range(Frame_number):
        tmp = Basic_img[:, :, frame]
        tmp[tmp <= 0] = 0
        tmp[tmp >= 255] = 255
        cv2.imwrite(filename+"_Basic"+ str(frame + 1)+".jpg", numpy.uint8(tmp))
    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency() 
    print ("The Processing time of the First step is %f s" % time)

    Final_img = BM3D_2nd_step(Basic_img,frames)
    for frame in range(Frame_number):
        tmp = Final_img[:,:,frame]
        tmp[tmp <= 0] = 0
        tmp[tmp >= 255] = 255
        cv2.imwrite(filename+"_Final"+str(frame+1)+".jpg", numpy.uint8(tmp))
    e3 = cv2.getTickCount()
    time = (e3 - e2) / cv2.getTickFrequency()
    print ("The Processing time of the Second step is %f s" % time)