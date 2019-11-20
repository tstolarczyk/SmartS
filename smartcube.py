
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import simple_norm
from ipywidgets import interact, fixed
import copy

theta  = 0 # view angles for simulated image & steady source

def twoD_Gaussian(x,y,amplitude,xo,yo,sigma_x,sigma_y,theta,offset):
    """
    define 2D-Gaussian function and pass independant variables x and y as a list
    """
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    
    return offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    
###########################################
def CreateSignalModelCube(LC,SourcePeak,xcen,ycen,xwidth,ywidth,nXYsize,sigma_vig,pixsize):

    # Create x and y indices
    x = np.linspace(0, nXYsize-1, nXYsize)
    y = np.linspace(0, nXYsize-1, nXYsize)
    x,y = np.meshgrid(x, y)
    
    # Create vignetting to be applied to image and transcient signal
    vign = twoD_Gaussian(x, y, 1, nXYsize/2,nXYsize/2, sigma_vig/pixsize, sigma_vig/pixsize, 0, 0)
    Source = twoD_Gaussian(x, y, 1, xcen, ycen, xwidth, ywidth, theta, 0)
    Source = SourcePeak * (Source/np.sum(Source))

    # Create Cube
    SignalModelCube = np.zeros((nXYsize,nXYsize,len(LC)))
    for rank,val in enumerate(LC):
        SignalModelCube[:,:,rank] = (Source*val)*vign 
    
    return SignalModelCube

###########################################
def CreateBkgModelCube(time,xwidth,ywidth,SteadySourcePeak,bkg,nXYsize,sigma_vig,pixsize):

    # Create x and y indices
    x = np.linspace(0, nXYsize-1, nXYsize)
    y = np.linspace(0, nXYsize-1, nXYsize)
    x,y = np.meshgrid(x, y)
    
    # Create vignetting to be applied to image and steadysource
    vign = twoD_Gaussian(x, y, 1, nXYsize/2,nXYsize/2, sigma_vig/pixsize, sigma_vig/pixsize, 0, 0)   
    SteadySource = twoD_Gaussian(x, y, 1, 55, 55, xwidth, ywidth, theta, 0)
    SteadySource = SteadySourcePeak * (SteadySource/np.sum(SteadySource))
    
    # Create Cube 
    BkgModelCube = np.zeros((nXYsize,nXYsize,len(time))) 
    for t in range(0,len(time)):
        BkgModelCube[:,:,t] = (bkg+SteadySource)*vign   
    
    return BkgModelCube

###########################################
def Old_CreateToyCubeImage(LC,SourcePeak,xcen,ycen,xwidth,ywidth,SteadySourcePeak,bkg,nXYsize,sigma_vig,pixsize):
    """
    Creates the cube (X,Y,T) toy model.
    Parameters
    ---------- 
    SourcePeak : float
        Max peak intensity of the signal
    LC   : array
        Time profile. Must be normalized to 1.
    xcen : int
        X center of bursting### Image characteristics
sigmaX = float(sigma_PSF/pixsize)
sigmaY = float(sigma_PSF/pixsize)### Image characteristics
sigmaX = float(sigma_PSF/pixsize)
sigmaY = float(sigma_PSF/pixsize) source
    ycen : int
        Y center of bursting source
    bkg : float
        Constant value for the mean background 
    Returns
    -------
    cube : array
        the toy model cube
    """
    
# Create x and y indices
    x = np.linspace(0, nXYsize-1, nXYsize)
    y = np.linspace(0, nXYsize-1, nXYsize)
    x,y = np.meshgrid(x, y)
    
# Create vignetting to be applied to image
    vign = twoD_Gaussian(x, y, 1, nXYsize/2,nXYsize/2, sigma_vig/pixsize, sigma_vig/pixsize, 0, 0)   

# Create signal image
    Source  = twoD_Gaussian(x, y, SourcePeak, xcen, ycen, xwidth, ywidth, theta, 0) 
    SteadySource = twoD_Gaussian(x, y, SteadySourcePeak, 55, 55, xwidth, ywidth, theta, 0)

# Create Cube
    cube = np.zeros((nXYsize,nXYsize,len(LC))) # Initialize the cube
    for k,t in enumerate(LC):
        cube[:,:,k]=((Source*t)+bkg+SteadySource)*vign  #Adding a 2nd constant source and * vignetting 
    return cube

###########################################
def Old_CreateBackgroundCube(LC,xwidth,ywidth,SteadySourcePeak,background,n2Dimsize,sigma_vignet,pixels_size):
    """
    Generating bkg model cube by summing all 30 Tbins with transient amplitude to zero
    and outputing a cube where each Tbin is identical (== sum_cube/30).
    Parameters
    ----------
    LC: array
        time profile (light curve) of transient. Here used only for the number of Tbins
    """
    cube_bkg_toy = CreateToyCubeImage(LC,0,0,0,xwidth,ywidth,SteadySourcePeak,background,n2Dimsize,sigma_vignet,\
    pixels_size) #Generating transient cube with transient signal to zero, and default background mean value ; Sigmas = 1 cause it will be a denominator
    # adding noise    
    cube_bkg = np.random.poisson(cube_bkg_toy)
    
    #sumbkg=np.sum(cube_bkg,axis=2)
    #size=np.shape(cube_bkg)[2]
    #cube_bkg=np.zeros(np.shape(cube_bkg))
    #for k in range(0,size): 
    #    cube_bkg[:,:,k]=sumbkg/size
    
    return cube_bkg

############################################
def CumulCube(Cube):
    """
    Create the cumulative cube from a cube.\
    Returns : Stacked_Cube
    """
    CumulCube = np.nancumsum(Cube,axis=2)
    
    return CumulCube

############################################
def CubeLC(Cube):
    """
    Create the lightcurve corresponding to a cube.\
    Returns : LC
    """
    LC = np.zeros(len(Cube[0,0,:]))
    
    for t in range(0,len(LC)):
        LC[t] = np.nansum(Cube[:,:,t][~np.isinf(Cube[:,:,t])])
        #print(LC)
    return LC

############################################
def ImageDistance(x1,y1,x2,y2):
    
    dist = np.abs(np.sqrt((x2-x1)**2 + (y2-y1)**2))
    return dist

############################################
def CubeDistance(x1,y1,z1,x2,y2,z2):
    
    dist = np.abs(np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2))
    return dist

############################################
def NoiseCube(cube_toy):
    """
    Add poissonian noise in a cube.\
    Returns : noisy_cube
    """
    np.random.seed() # creating a new seed. Otherwise get same seed in each multiprocessor pool
    cube_noisy = np.float64(np.random.poisson(cube_toy))
    
    return cube_noisy

###########################################
def Old_CubeLightCurve(cube):
    """
    Project the cube on the time dimension
    """
    LC = np.nansum(cube,axis=(0,1)) 
    
    return LC

###########################################
def StackCubeToImage(cube):
    """
    Stack a cube in one image containg the mean value in the third axe (time for now)
    """
    sum_cube = np.sum(cube,axis=2)
    #size = np.shape(cube)[2]
    #stack_cube = sum_cube/size
    
    return sum_cube

###########################################
def print_peaks(message,data,timebin):
    
    if data.ndim == 1 : tp1d,A1d = FindLightCurveMax(data)
    if data.ndim == 3 : xp,yp,tp3d,A3d,tp1d,A1d = get_peaks(data)
    
    print(message)
    print("------------------")
    print("       - light curve maximum - amplitude (Cts/Tbin)      = ",A1d)
    print("                             - time      (slice)         = ",tp1d)
    print("                             - time      (s)             = ",tp1d*timebin)
    
          
    if data.ndim == 1 : 
        print("------------------\n")
        return tp1d,A1d
        
    print("\n       - Cube maximum        - amplitude (Cts/Tbin/pix)  = ",A3d)
    print("                             - time      (slice)         = ",tp3d)
    print("                             - time      (s)             = ",tp3d*timebin)
    print("                             - x,y       (px)            = ",xp,yp)
    print("------------------\n")
    return xp,yp,tp3d,A3d,tp1d,A1d


###########################################
# This be splitted in two : a search of max in a cube and a search of max in a light curve
def get_peaks(cube):
    """
    Reads in a cube and returns X,Y,T peak value of a transient
    input: cube
    output: 1D, 3D, smoothed Tpeaks and X,Y
    """
    # Create the lightcurve form the cube - get the lightcurve time of maximum and amplitude at maximum
    tp1d, A1d = FindLightCurveMax(np.sum(cube,axis=(0,1)))
    yp,xp,tp3d,A3d = FindCubeMax(cube)
    return xp,yp,tp3d,A3d,tp1d,A1d

###########################################
def FindLightCurveMax(LightCurve):
    tmax=np.argmax(LightCurve) 
    Amax = np.max(LightCurve)
    return tmax,Amax

###########################################
def FindCubeMax(cube):
    #cube_temp=copy.deepcopy(cube)
    #cube_temp[:,:,-1]=0 ; cube_temp[:,:,-2]=0 ; cube_temp[:,:,0:2]=0 # Hack to avoid picking max from first two or last two slices due to wavelets bounce
    
    xmax,ymax,tmax=np.unravel_index(np.argmax(cube),np.shape(cube)) # max in the cube
    Amax = np.max(cube) # to be computed

    return xmax,ymax,tmax,Amax

###########################################
def Old_FindCubeMax(cube):
    cube[:,:,-1]=0 ; cube[:,:,-2]=0 ; cube[:,:,0:2]=0 # Hack to avoid picking max from first two or last two slices due to wavelets bounce
    xmax,ymax,tmax=np.unravel_index(np.argmax(cube),np.shape(cube)) # max in the cube
    Amax = np.max(cube) # to be computed
    return xmax,ymax,tmax,Amax

###########################################
def WriteCubeToFile(cube,filename,hdr):
    """
    Creates the cube ".fits" file with header (header = 2Darray [values' names, values])
    """
    fits.writeto(filename,np.asarray(cube.T,dtype=np.float64),overwrite=True)
    
    if hdr != '': 
        for i in range(0,len(hdr[0])): 
            fits.setval(filename, hdr[0,i], value=hdr[1,i])
    
    return

###########################################
def AddCubeToFile(filename,cube,namecube,hdr):
    
    HDU_list = fits.open(filename)
    new_HDU = fits.ImageHDU(cube.T,name=namecube,header=hdr)
    
    try:
        HDU_list[new_HDU.name]
        del HDU_list[new_HDU.name]
    except:pass

    HDU_list.append(new_HDU)
    HDU_list.writeto(filename,overwrite=True)
    
    return

###########################################
def WriteTopFile(filename,text):
    """
    Write at the top of a text file
    """
    with open(filename, "r+") as f: s = f.read(); f.seek(0); f.write(text + s)
    return

###########################################
def ReadCubeFromFile(filename,fich_type):
    """
    Read the cube file (type : fits) 
    """
    if fich_type=="fits":
        cube,header = fits.getdata(filename,header=True)
    return cube.T,header

############################################
def SepSourceBkgImage(Image,x_src,y_src,Xsigma,Ysigma,fact):
    x = np.arange(0,len(Image[:,0]))
    y = np.arange(0,len(Image[0,:]))
    
    mask_out = np.sqrt((x[np.newaxis,:]-x_src)**2 + (y[:,np.newaxis]-y_src)**2) > fact*np.sqrt(Xsigma**2+Ysigma**2)
    mask_in = np.sqrt((x[np.newaxis,:]-x_src)**2 + (y[:,np.newaxis]-y_src)**2) <= fact*np.sqrt(Xsigma**2+Ysigma**2)
    
    Image_Src = copy.deepcopy(Image) ; Image_Bkg = copy.deepcopy(Image)
    
    Image_Src[mask_out] = float('NaN')
    Image_Bkg[mask_in] = float('NaN')
        
    return Image_Bkg,Image_Src

############################################
def SepSourceBkgCube(Cube,x,y,sigx,sigy,fact):
    """
    Uses SepSourceBkgImage for each slices of a cube.\
    Returns : cube_bkg,cube_Sbkg
    """
    timesize = np.shape(Cube)[2]
    Cube_bkg  = copy.deepcopy(Cube)
    Cube_Sbkg = copy.deepcopy(Cube)
    
    for k in range(0,timesize):
        Cube_bkg[:,:,k],Cube_Sbkg[:,:,k] = SepSourceBkgImage(Cube[:,:,k],x,y,sigx,sigy,fact)
    
    return Cube_bkg,Cube_Sbkg

############################################
def MakeRingCube(Cube,xcen,ycen,Fact_in,Fact_out,SigX,SigY):
    
    RingCube = np.zeros(np.shape(Cube))
    for t in range(0,len(Cube[0,0,:])):
        RingCube[:,:,t] = MakeRingImage(Cube[:,:,t],xcen,ycen,Fact_in,Fact_out,SigX,SigY)
    
    return RingCube

############################################
def MakeRingImage(Image,xcen,ycen,Fact_in,Fact_out,SigX,SigY):
    
    x = np.arange(0,len(Image[:,0]))
    y = np.arange(0,len(Image[0,:]))
    
    mask_in  = np.sqrt((x[np.newaxis,:]-xcen)**2 + (y[:,np.newaxis]-ycen)**2) < Fact_in*np.sqrt(SigX**2+SigY**2)
    mask_off = np.sqrt((x[np.newaxis,:]-xcen)**2 + (y[:,np.newaxis]-ycen)**2) >= Fact_out*np.sqrt(SigX**2+SigY**2)
    
    ImageRing = copy.deepcopy(Image)
    ImageRing[mask_in] = float('NaN') ; ImageRing[mask_off] = float('NaN')
    
    return ImageRing

###########################################
def PlotCubeInOneImage(cube,title):
    """
    Plots sqrt of the sum of each images of the cube over time 
    """
    sum_cube=np.sum(cube,axis=2)
    plt.imshow(np.sqrt(sum_cube))
    plt.title(title)
    plt.colorbar()
    plt.grid(True)
    plt.show()
    return

###########################################
def PlotCubeImage(dbg,message,cube_image,timearr,t_diplay,Tbin):
    """
    2 Plots for the Cube (X,Y,T) : 1) Image at T time : Y(X,T)
                                   2) Light curve in function of T : LC(T)
    Parameters
    ----------
    cube_image : array
        2D * 1D Cube ((x*y) * (time))
    t_display : int
        Slice of time in the cube
    """
    if (dbg==0):
        return
    print(message)
    if (dbg==1):
        PlotCube(cube_image,timearr,t_diplay)
    if (dbg==2):
        interact(PlotCube,cube=fixed(cube_image),time=fixed(timearr),tdisplay=(int(timearr[0]),int(timearr[-1]),Tbin))
    return

###########################################
def PlotCube(cube,time,tdisplay):
    """
    Plot the image of the cube at tdisplay, as well as the cube projected lightcurve, with a dshed line at tdisplay
    """
    #Tsize=np.shape(cube)[-1]
    #print("No elements =",Tsize)
    #time = np.arange(0,Tsize)*timebin # timebin is real time of exposure
    
    im_t = cube[:,:,np.where(time==tdisplay)[0][0]].astype(float) # Create image at tdisplay
    LC   = CubeLC(cube)

    fig = plt.figure(figsize=(10,5))    
    # Image at tdisplay from the cube
    left, bottom, width, height = 0.05, 0., 0.45, 1
    ax1 = fig.add_axes([left, bottom, width, height])
    ax1.grid(False)
    stretch='sqrt'
    
    if np.isnan(cube).any() : 
        new_im_t = np.ma.array(im_t, mask=np.isnan(im_t)) # Array with masked values
        cmap = plt.cm.jet
        cmap.set_bad('white',1.)
        p1=ax1.imshow(new_im_t, cmap=cmap,vmax=np.max(cube[~np.isnan(cube)]))
    
    else : p1=ax1.imshow(im_t,norm=simple_norm(im_t, stretch),vmax=np.max(cube))
        
    cbar=plt.colorbar(p1,fraction=0.045,ax=ax1)
    ax1.set_xlabel('x (px)')
    ax1.set_ylabel('y (px)')
    
    # Lightcurve from the cube
    left, bottom, width, height = 0.65, 0.13, 0.5, 0.8
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.axvline(tdisplay,color='blue',ls=':')
    ax2.set_xlabel('Time (sec)',fontsize=15)
    ax2.set_ylabel('Counts',fontsize=15)
    ax2.plot(time,LC)
    
    plt.show()
    
    return

###########################################
def OldPlotCube(cube, tdisplay,timebin):
    """
    Plot the image of the cube at tdisplay, as well as the cube projected lightcurve, with a dshed line at tdisplay
    """
    Tsize=np.shape(cube)[2]
    print("No elements =",Tsize)
    
    time = np.arange(0,Tsize)*timebin # timebin is real time of exposure
    
    im_t = cube[:,:,tdisplay].astype(float) # Create image at tdisplay
    LC = np.sum(cube,axis=(0,1)) # Project the curbe on the time dimension

    fig = plt.figure(figsize=(10,5))    
    # Image at tdisplay from the cube
    left, bottom, width, height = 0.05, 0., 0.45, 1
    ax1 = fig.add_axes([left, bottom, width, height])
    ax1.grid(False)
    stretch='sqrt'
    p1=ax1.imshow(im_t,norm=simple_norm(im_t, stretch),vmax=np.max(cube))
    cbar=plt.colorbar(p1,fraction=0.045,ax=ax1)
    
    # Lightcurve from the cube
    left, bottom, width, height = 0.65, 0.13, 0.5, 0.8
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.axvline(tdisplay*timebin,color='blue',ls=':')
    ax2.set_xlabel('Time (sec)',fontsize=15)
    ax2.set_ylabel('Counts',fontsize=15)
    ax2.plot(time,LC)
    
    plt.show()
    
    return

###########################################
def PlotImage(image,title):
    """
    Plots image (could be a slacked cube) 
    """
    plt.imshow(image)
    plt.title(title)
    plt.colorbar()
    plt.grid(False)
    plt.show()
    return