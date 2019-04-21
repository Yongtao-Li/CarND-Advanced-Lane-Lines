## Writeup for Advanced Lane Lines

## by Yongtao Li

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted_camera_calibration/calibration4.jpg "distorted vs undistorted"
[image2]: ./test_images/test6.jpg "original"
[image3]: ./output_images/undistorted_test_images/test6.jpg "undistorted"
[image4]: ./output_images/threshholded_test_images/test6.jpg "thresholded"
[image5]: ./output_images/undistorted_test_images/straight_lines2.jpg "undistorted_straight"
[image6]: ./output_images/warped_test_images/straight_lines2.jpg "warped_straight"
[image7]: ./output_images/warped_test_images/test6.jpg "warped"
[image8]: ./output_images/fitted_test_images/test6.jpg "fitted"
[image9]: ./output_images/final_test_images/test6.jpg "final"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

##### please see the code from the camera calibration session in my notebook "P2_YL.ipynb". I will explain the key code as following.

First of all, I have iterated through all images and find corners on the chessboard.

```python
ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
```

Then with all image points from corners identified and object points generated at z = 0 plane, I could calcuate the calibration matrix and distortion coefficients. By using these matrix and coefficient, I have undistorted all the previous calibration images.

```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
dst = cv2.undistort(img, mtx, dist, None, mtx)
```

The following image is one of the exmaples. Left side picture is before undistortion and right side is after undistortion.
![alt text][image1]

I have also preserved the calibration matrix and distortion coefficients as a pickle file for the following pipeline usage.

```python
dist_pickle['mtx'] = mtx
dist_pickle['dist'] = dist
pickle.dump(dist_pickle, open('camera_cal/dist_pickle.p', 'wb'))
```

### Pipeline (single images)

##### all the help functions that are used in the image and video pipeline are located in my notebook "P2_YL.ipynb" under the help function session.

#### 1. Provide an example of a distortion-corrected image.

By using previous prepared pickle file, I iterate through all test images and undistort all of them by using previous calibration matrix and distortion coefficients.

```python
# function to undistort image using previous calibration
def undistort(img, pfile):
    
    # read in camera matrix and distortion coefficients
    dist_pickle = pickle.load(open(pfile, 'rb'))
    mtx = dist_pickle['mtx']
    dist = dist_pickle['dist']
    
    return cv2.undistort(img, mtx, dist, None, mtx)
```

Here is an example before and after undistrotion.

original image             |  undistorted image
:-------------------------:|:-------------------------:
![alt text][image2]       |  ![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

For color transforms, I have used HLS color space and apply thresholds to S channel to identify lane lines. I have also used the gradients in x, y direction, combined with magnitude of gradients and direction of gradients. The thresholds are actually not very strict, because the idea is to identify lane line pixels under different light and shade conditions. Once we apply perspective transform, we'll see pretty much only lane line left on the bird's view image.

```python
# function to threshold images by color space and gradient
def threshholded(img, s_thresh=(150, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=15, thresh=(20,100))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=15, thresh=(20,100))
    mag_binary = mag_thresh(img, sobel_kernel=15, mag_thresh=(20,100))
    dir_binary = dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.4))
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # combine binaries into one    
    combined_binary = np.zeros_like(s_binary)
    combined_binary[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1) ] = 1
    
    return combined_binary
```

All thresholds are working together and here is an example of undistorted image before and after applying thresholds to identy lane lines.

undistorted image          |  thresholded image
:-------------------------:|:-------------------------:
![alt text][image3]        |  ![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

First of all I have tried my perspective transform on straight line test images, in order to get a good set of source and destination points.
```python
# function for perspective transform
def warp(img):
    # define calibration box in source (original) and destination (desired or warped) coordinates
    img_size = (img.shape[1], img.shape[0])
    # four source coordinates
    src = np.float32(
    [
        [568, 470],
        [260, 680],
        [717, 470],
        [1043, 680]
    ])
    # four desired coordinates
    dst = np.float32(
    [
        [200, 0],
        [200, 680],
        [1000, 0],
        [1000, 680]
    ])
    src_pts = src.astype(int).reshape((-1,1,2))
    dst_pts = dst.astype(int).reshape((-1,1,2))
    # compute the perspective transform, M
    M = cv2.getPerspectiveTransform(src, dst)
    # create warped image - use linear interpolation
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return img, warped
```

undistorted image          |  warped image
:-------------------------:|:-------------------------:
![alt text][image5]        |  ![alt text][image6]

Once I'm statisfied with the straight line results, I have applied the perspective transform to all test images, as you could see in the following example. I have also simply swapped the source and destination points to get an unwarp function that will be used later for mapping lane lines back on original image.

undistorted image          |  warped image
:-------------------------:|:-------------------------:
![alt text][image3]        |  ![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Please see the funciton "find_lane_pixels" and function "fit_polynomial". I have used the sliding window method as you could see from the following image. The first window starts from the bottom where a histogram help identify lane line from previous warped image. Then the window goes up as it find more pixels for lane lines and adjusts its center. Once all lane line pixels identified, I use numpy to fit a polynomial using all these pixels.

```python
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```

warped image               |  fitted image
:-------------------------:|:-------------------------:
![alt text][image7]        |  ![alt text][image8]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Here is the key code to calculate radius of curvature for both lane lines and the offset of the car with respect to center. Next I'll have these information printed out on the original image.

```python
# calculate the curvature
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# We'll choose the maximum y-value, corresponding to the bottom of the image
y_eval = np.max(ploty)

##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
left_fit_cr = np.zeros_like(left_fit)
right_fit_cr = np.zeros_like(right_fit)
left_fit_cr[0] = left_fit[0] * xm_per_pix / (ym_per_pix) ** 2
right_fit_cr[0] = right_fit[0] * xm_per_pix / (ym_per_pix) ** 2
left_fit_cr[1] = left_fit[1] * xm_per_pix / ym_per_pix
right_fit_cr[1] = right_fit[1] * xm_per_pix / ym_per_pix
left_curverad = (1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1] ) ** 2) ** 1.5 / np.absolute(2*left_fit_cr[0])  ## Implement the calculation of the left line here
right_curverad = (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1] ) ** 2) ** 1.5 / np.absolute(2*right_fit_cr[0]) ## Implement the calculation of the right line here

# calculate the offset with respect to center
lane_ctr =  ( (left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]) +(right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]) ) * 0.5
offset = ( out_img.shape[1] * 0.5 - lane_ctr ) * xm_per_pix
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Once I have a good polynomial fit on both lane lines, I put these lines on a blank image and shade the area green. Then I use previous unwarp function to reverse perspective transform and combine it with the undistorted image. Also I have put text information for the radius of curvature and offset on the top of the final image.

```python
# create an image to draw the lines on
warp_zero = np.zeros_like(img_binary_warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = unwarp(color_warp)
#plt.imshow(newwarp)

# Combine the result with the original image
img_undist = mpimg.imread('output_images/undistorted_test_images/' + os.path.split(fname)[1])
result = cv2.addWeighted(img_undist, 1, newwarp, 0.3, 0)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(result,'left curvature is %8.3fm; right curvature is %8.3fm' % (left_curverad, right_curverad),(10,100), font, 1,(255,255,255),2,cv2.LINE_AA)
if offset_ctr < 0:
    cv2.putText(result,'vehicle is %8.3f m left of center' % abs(offset_ctr),(10,200), font, 1,(255,255,255),2,cv2.LINE_AA)
elif offset_ctr > 0:
    cv2.putText(result,'vehicle is %8.3f m right of center' % abs(offset_ctr),(10,200), font, 1,(255,255,255),2,cv2.LINE_AA)
else:
    cv2.putText(result,'vehicle is centered!',(10,200), font, 1,(255,255,255),2,cv2.LINE_AA)
```

undistorted image          |  final image
:-------------------------:|:-------------------------:
![alt text][image3]        |  ![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my project video result](./output_videos/project_video.mp4). As you could see, my video pipleline works pretty well on project video and goes pretty smooth between different frames.

I have also tried my pipeline on [the challenge video](./output_videos/challenge_video.mp4) and [harder challenge video](./output_videos/harder_challenge_video.mp4). The video pipeline has done a decent job on the challenge video where the car goes under a bridge and there is also pavement divide on the road which could be misidentified as lane line. In order to make it work for challenge videos, I have made a few enhancements on the pipeline:

* use global line object to track previous lane line fit results

* use a region around previous fit lines for next frame if previous detection is successful. Please see "search_around_poly" function for more details.

* use a weighted average to calculate best fit over n succcessful iterations

```python
left_line.best_fit = np.add(np.multiply(left_line.best_fit, n), left_line.current_fit) / (n+1)
```

* use the derative of the fitted line and a threshold to reject obvious wrong detection

```python
if abs(2*left_fit[0]*360+left_fit[1])>=0.55 or abs(2*right_fit[0]*360+right_fit[1]) >=0.55:
    left_line.detected = False
    right_line.detected = False
else:
    left_line.detected = True
    right_line.detected = True
```

* use the previous best fit coefficients if the current fit is rejected

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Overall my approach I took is pretty straight forward and works pretty well on the project video and also does a decent job on challenge video. The pipeline takes steps through camera calibration, image undistortion, lane line pixel filter, perspective transform, sliding window detection and lane line fitting. As I mentioned earlier, adding the tracking, smoothing, looking from prior fit and sanity check is not really necessary for completing project videos, but they are definitely needed for the challege video. The harder challege video would work okay on the beginning, but there is a sharp turn later where only one lane line is visible. Further improvement could be made to handle this corner case better, perhaps by tracking left and right lane lines individually.
