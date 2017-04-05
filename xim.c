// xim : basic framework to handle XIMEA cameras with RPIt
// Compilation: make
// OpenCV and XIAPI should be installed.
// Performance tuning:
//	- Force cpu performance mode:
//		Add to "/etc/rc.local":
//		for cpucore in /sys/devices/system/cpu/cpu?; do echo performance | sudo tee $cpucore/cpufreq/scaling_governor > /dev/null; done
// 	- Avoid USB suspend :
//		Modify "/etc/default/grub":
//		GRUB_CMDLINE_LINUX_DEFAULT="quiet splash usbcore.autosuspend=-1"
//		sudo update-grub
//	- Disable memory limit for USB:
//		Add to "/etc/rc.local":
//		echo 0 > /sys/module/usbcore/parameters/usbfs_memory_mb
//	- Allow user access:
//		gpasswd -a "$(whoami)" plugdev
//	- Allow user to use realtime priorities:
//		sudo groupadd -fr realtime
//		echo '*         - rtprio   0' | sudo tee    /etc/security/limits.d/ximea.conf > /dev/null
//		echo '@realtime - rtprio  81' | sudo tee -a /etc/security/limits.d/ximea.conf > /dev/null
//		echo '*         - nice     0' | sudo tee -a /etc/security/limits.d/ximea.conf > /dev/null
//		echo '@realtime - nice   -16' | sudo tee -a /etc/security/limits.d/ximea.conf > /dev/null
//		sudo gpasswd -a "$(whoami)" realtime
//
// JG, 30.03.17

#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <termios.h>
#include <memory.h>
#include <m3api/xiApi.h>
#include <iostream>
#include <opencv2/core.hpp> 
#include <opencv2/features2d.hpp>
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>

#define HandleResult(res,place) if (res!=XI_OK) {printf("Error after %s (%d).\n",place,res);goto finish;}

#define XIM_LIVE_VIDEO										// Live video on/off
#define XIM_XIAPI_BUFFERS		3							// Number of frame buffers
#define XIM_VIDEO_FSKIP			20						// Live video frame subsampling
#define XIM_EXPOSURE				1800					// us
#define XIM_TIMEOUT					5000					// ms
#define XIM_VIDEO_NAME			"Live video"	// Name of the live video window

#define XIM_NB_FEATURES			4							// NB of features to be extracted
#define XIM_ROI_MARGIN			40.0					// Margin in percent around a feature

// Blob detector parameters: 
// see http://docs.opencv.org/3.0-beta/modules/features2d/doc/common_interfaces_of_feature_detectors.html#simpleblobdetector
#define XIM_BLOB_MIN_THRES	150.0
#define XIM_BLOB_MAX_THRES	151.0
#define XIM_BLOB_THRES_STEP	1.0
#define XIM_BLOB_MIN_DIST		200.0
#define XIM_BLOB_COLOR			255
#define XIM_BLOB_MIN_AREA		150.0
#define XIM_BLOB_MAX_AREA		8000.0
//#define XIM_BLOB_MIN_CIRC		0.5
//#define XIM_BLOB_MAX_CIRC		1.0
//#define XIM_BLOB_MIN_CONVEX	0.87
//#define XIM_BLOB_MAX_CONVEX	1.0
//#define XIM_BLOB_MIN_INER_R	0.01
//#define XIM_BLOB_MAX_INER_R	0.1

using namespace cv;
using namespace std;

// 
// getch and kbhit
// 
// Helper functions for interactions in console.
// 
 
int getch(void)
{
  struct termios oldt, newt;
  int ch;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  ch = getchar();
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  return ch;
}

int kbhit(void)
{
  struct termios oldt, newt;
  int ch;
  int oldf;

  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
  fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

  ch = getchar();

  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  fcntl(STDIN_FILENO, F_SETFL, oldf);

  if (ch != EOF)
  {
    ungetc(ch, stdin);
    return 1;
  }

  return 0;
}

// Compute padding
int xim_get_padding_x( XI_IMG* xiimg )
{
	switch( xiimg->frm )
	{
	case XI_RAW16:
	case XI_MONO16:
		return xiimg->padding_x/2;
	default:
		return xiimg->padding_x;	
	}	
}

// Converts a XiAPI image to OpenCV IplImage.
// Creates CV image.
// Return NULL if error, valid pointer otherwise.
IplImage* xim_xiimg2cvipl( XI_IMG* xiimg )	{
	IplImage* cvipl;
	
	// Create new IPl Image according to XI_IMG characteristics
	switch( xiimg->frm )
		{
		case XI_MONO8       : cvipl = cvCreateImage( cvSize( xiimg->width + xim_get_padding_x( xiimg ), xiimg->height ), IPL_DEPTH_8U, 1 ); break;
		case XI_RAW8        : cvipl = cvCreateImage( cvSize( xiimg->width, xiimg->height ), IPL_DEPTH_8U, 1 ); break;		
		case XI_MONO16      :
		case XI_RAW16       : cvipl = cvCreateImage( cvSize( xiimg->width, xiimg->height ), IPL_DEPTH_16U, 1 ); break;
		case XI_RGB24       :
		case XI_RGB_PLANAR  : cvipl = cvCreateImage( cvSize( xiimg->width, xiimg->height ), IPL_DEPTH_8U, 3 );	break;
		case XI_RGB32       : cvipl = cvCreateImage( cvSize( xiimg->width, xiimg->height ), IPL_DEPTH_8U, 4 ); break;
		default :
			printf( "xim_xiimg2cvipl error: unknown format.\n");
			return NULL;
		}
	
	// Defines the pointer to image data
	cvipl->imageData=(char*)xiimg->bp;

	return cvipl;
}

int main( int argc, char* argv[] )
{
	XI_IMG 													image;
	IplImage*												cv_image;
	cv::Mat													cv_mat, cv_im_bin;
	#ifdef XIM_LIVE_VIDEO
	cv::Mat													im_with_keypoints;
	unsigned long long							frame_cnt = 0;
	#endif
	cv::SimpleBlobDetector::Params 	params;
	cv::Ptr<cv::SimpleBlobDetector> detector;
	std::vector<KeyPoint> 					keypoints, keypoints_ROI;
	int															i;
	struct timespec 								instant_1, instant_2;
	unsigned long long							loop_duration = 0;
	unsigned char										detected = 0;
	std::vector<Rect>								xim_ROI( XIM_NB_FEATURES );
	std::vector<Mat>								xim_ROI_mat( XIM_NB_FEATURES );
	std::vector<Point2f>						xim_cog( XIM_NB_FEATURES );
	
	// Initialize image buffer
	memset( &image, 0, sizeof(image) );
	image.size = sizeof( XI_IMG );

	// XIMEA API V4.05
	HANDLE xiH = NULL;
	XI_RETURN stat = XI_OK;

	// Retrieving a handle to the camera device 
	printf( "Opening first camera.\n" );
	stat = xiOpenDevice( 0, &xiH );
	HandleResult( stat, "xiOpenDevice" );

	// Setting "exposure" parameter (us)
	stat = xiSetParamInt( xiH, XI_PRM_EXPOSURE, XIM_EXPOSURE );
	HandleResult( stat,"xiSetParam (exposure set)" );
	
	// Limit buffer size
	stat = xiSetParamInt( xiH, XI_PRM_BUFFERS_QUEUE_SIZE, XIM_XIAPI_BUFFERS );
	HandleResult( stat,"xiSetParam (buffers queue size)" );
	
	// GetImage retrieves the most recent image
	stat = xiSetParamInt( xiH, XI_PRM_RECENT_FRAME, 1 );
	HandleResult( stat,"xiSetParam (recent frame)" );
	
	// Starting acquisition
	printf( "Starting acquisition.\n" );
	stat = xiStartAcquisition( xiH );
	HandleResult( stat, "xiStartAcquisition" );

	// Get first image to detect format
	stat = xiGetImage( xiH, XIM_TIMEOUT, &image );
	HandleResult( stat, "xiGetImage" );
	
	// Protect all OpenCV code with a try...catch
	try	{
		// Initialize OpenCV
		cv_image = xim_xiimg2cvipl( &image );
		if ( cv_image == NULL )	{
			printf( "Error after xim_xiimg2cvipl (NULL).\n" );
			goto finish;
		}
		cv_mat = cv::cvarrToMat( cv_image );

		// Create video window
		#ifdef XIM_LIVE_VIDEO
		cv::namedWindow( XIM_VIDEO_NAME, WINDOW_AUTOSIZE );
		#endif

		// Define thresholds
		params.minThreshold = XIM_BLOB_MIN_THRES;
		params.maxThreshold = XIM_BLOB_MAX_THRES;
		params.thresholdStep = XIM_BLOB_THRES_STEP;
		
		// Define Repeatability
		params.minRepeatability = 0;
		
		// Define min distance
		params.minDistBetweenBlobs = XIM_BLOB_MIN_DIST;
		
		// Filter by color: 0-> dark blob; 255->bright blob
		#ifdef XIM_BLOB_COLOR
		params.filterByColor = true;
		params.blobColor = XIM_BLOB_COLOR;
		#else
		params.filterByColor = false;
		#endif

		// Filter by Area.
		#ifdef XIM_BLOB_MIN_AREA
		params.filterByArea = true;
		params.minArea = XIM_BLOB_MIN_AREA;
		#ifdef XIM_BLOB_MAX_AREA
		params.maxArea = XIM_BLOB_MAX_AREA;
		#endif
		#else
		params.filterByArea = false;
		#endif

		// Filter by Circularity
		#ifdef XIM_BLOB_MIN_CIRC
		params.filterByCircularity = true;
		params.minCircularity = XIM_BLOB_MIN_CIRC;
		#ifdef XIM_BLOB_MAX_CIRC
		params.maxCircularity = XIM_BLOB_MAX_CIRC;
		#endif
		#else
		params.filterByCircularity = false;
		#endif

		// Filter by Convexity
		#ifdef XIM_BLOB_MIN_CONVEX
		params.filterByConvexity = true;
		params.minConvexity = XIM_BLOB_MIN_CONVEX;
		#ifdef XIM_BLOB_MAX_CONVEX
		params.maxConvexity = XIM_BLOB_MAX_CONVEX;
		#endif
		#else
		params.filterByConvexity = false;
		#endif

		// Filter by Inertia
		#ifdef XIM_BLOB_MIN_INER_R
		params.filterByInertia = true;
		params.minInertiaRatio = XIM_BLOB_MIN_INER_R;
		#ifdef XIM_BLOB_MAX_INER_R
		params.maxInertiaRatio = XIM_BLOB_MAX_INER_R;
		#endif
		#else
		params.filterByInertia = false;	
		#endif
		
		// Create blob detector
		detector = cv::SimpleBlobDetector::create( params );
		
		// Get current time
		clock_gettime( CLOCK_MONOTONIC, &instant_1 );
			
		// Acquisition loop
		while( 1 )
		{
			// Getting image from camera
			stat = xiGetImage( xiH, XIM_TIMEOUT, &image );
			HandleResult( stat, "xiGetImage" );
			
			// Get current time
			clock_gettime( CLOCK_MONOTONIC, &instant_2 );
			
			// Caluclate loop duration (us)
			loop_duration = 	( instant_2.tv_sec - instant_1.tv_sec ) * 1000000 +
												( instant_2.tv_nsec - instant_1.tv_nsec ) / 1000;
			printf( "Loop duration: %llu us\n", loop_duration );
			
			// Switch variable for next iteration
			memcpy( &instant_1, &instant_2, sizeof( struct timespec ) );
			
			// Update CV image
			cv_image->imageData=(char*)image.bp;
			cv_mat = cv::cvarrToMat( cv_image );

			// Detect blobs only if needed (takes time)
			if ( !detected )
			{
				// Image binarization
				cv::threshold( cv_mat, cv_im_bin, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU );
				// Global blob detection
				detector->detect( cv_im_bin, keypoints );
				if ( keypoints.size() == XIM_NB_FEATURES )	{
					printf( "Detect blobs: all features detected.\n" );
					// Initializing ROIs
					for ( i = 0; i < XIM_NB_FEATURES; i++ )	{
						xim_ROI[i].x = keypoints[i].pt.x - ( keypoints[i].size * ( 1.0 + XIM_ROI_MARGIN / 100.0 ) ) / 2.0;
						xim_ROI[i].y = keypoints[i].pt.y - ( keypoints[i].size * ( 1.0 + XIM_ROI_MARGIN / 100.0 ) ) / 2.0;
						xim_ROI[i].width = keypoints[i].size * ( 1.0 + XIM_ROI_MARGIN / 100.0 );
						xim_ROI[i].height = keypoints[i].size * ( 1.0 + XIM_ROI_MARGIN / 100.0 );
						
						// ROI saturation
						if ( xim_ROI[i].x < 0 )
							xim_ROI[i].x = 0;
						if ( xim_ROI[i].y < 0 )
							xim_ROI[i].y = 0;
						if ( xim_ROI[i].x + xim_ROI[i].width > cv_mat.cols )
							xim_ROI[i].width = cv_mat.cols - xim_ROI[i].x;
						if ( xim_ROI[i].y + xim_ROI[i].height > cv_mat.rows )
							xim_ROI[i].height = cv_mat.rows - xim_ROI[i].y;
					}
					
					detected = 1;
				}
				else
					printf( "Detect blobs: missing features.\n" );
			}
			
			// If features are localized, switch to tracking mode
			if ( detected )
			{
				printf( "Feature tracking mode.\n" );
				
				// Detect blob in ROIs
				for ( i = 0; i < XIM_NB_FEATURES; i++ )	{
					
					// Extract ROI from acquired image
					xim_ROI_mat[i] = cv_mat( xim_ROI[i] );
					
					// ROI binarization
					cv::threshold( xim_ROI_mat[i], cv_im_bin, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU );
					
					// Blob detection in ROI
					detector->detect( cv_im_bin, keypoints_ROI );
					
					// Error handling
					if ( keypoints_ROI.size() == 0 )	{
						printf( "No feature in ROI #%d!\n", i );
						detected = 0;
						break;
					}
					if ( keypoints_ROI.size() > 1 )	{
						printf( "More than 1 feature in ROI #%d!\n", i );
					}
					
					// Update keypoint
					keypoints[i] = keypoints_ROI[0];
					
					// Update ROI
					keypoints[i].pt.x += xim_ROI[i].x;
					keypoints[i].pt.y += xim_ROI[i].y;
					xim_ROI[i].x = keypoints[i].pt.x - ( keypoints[i].size * ( 1.0 + XIM_ROI_MARGIN / 100.0 ) ) / 2.0;
					xim_ROI[i].y = keypoints[i].pt.y - ( keypoints[i].size * ( 1.0 + XIM_ROI_MARGIN / 100.0 ) ) / 2.0;
					xim_ROI[i].width = keypoints[i].size * ( 1.0 + XIM_ROI_MARGIN / 100.0 );
					xim_ROI[i].height = keypoints[i].size * ( 1.0 + XIM_ROI_MARGIN / 100.0 );
					
					// ROI saturation
					if ( xim_ROI[i].x < 0 )
						xim_ROI[i].x = 0;
					if ( xim_ROI[i].y < 0 )
						xim_ROI[i].y = 0;
					if ( xim_ROI[i].x + xim_ROI[i].width > cv_mat.cols )
						xim_ROI[i].width = cv_mat.cols - xim_ROI[i].x;
					if ( xim_ROI[i].y + xim_ROI[i].height > cv_mat.rows )
						xim_ROI[i].height = cv_mat.rows - xim_ROI[i].y;
						
					// Compute feature center of gravity
					xim_cog[i].x = keypoints[i].pt.x;
					xim_cog[i].y = keypoints[i].pt.y;
					
				}
			}
			
			#ifdef XIM_LIVE_VIDEO
			// Display image at desired fps
			if ( !( frame_cnt++ % XIM_VIDEO_FSKIP ) )	{
		
				// Add keypoints to current image
				drawKeypoints( cv_mat, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
				if ( detected )	{
					for ( i = 0; i < XIM_NB_FEATURES; i++ )	{
						cv::rectangle( im_with_keypoints, xim_ROI[i].tl(), xim_ROI[i].br(), Scalar(0,255,0), 2, 8, 0 );
						cv::circle( im_with_keypoints, xim_cog[i], 2, Scalar(255,0,0), 1, 8, 0 );
					}
				}
				
				// Draw image
				cv::imshow( XIM_VIDEO_NAME, im_with_keypoints );
				
				// Exit if key pressed + let the time for the image to be displayed
				if ( cvWaitKey( XIM_EXPOSURE / 1000 ) != -1 )
					break;
				}
			#else
			// Exit if key pressed
			if ( kbhit( ) )
				break;
			#endif
		}
	}
	
	catch( cv::Exception& e )
	{
    const char* err_msg = e.what();
    std::cout << "Exception caught: " << err_msg << std::endl;
	}
	
	// Shortcut in case of error
	finish:
	
	// Cleanup
	printf( "Stopping acquisition.\n" );
	#ifdef XIM_LIVE_VIDEO
	destroyWindow( XIM_VIDEO_NAME );
	#endif
	cvReleaseImage(	&cv_image	);
	xiStopAcquisition( xiH );
	xiCloseDevice( xiH );

	printf("Done.\n");

	return 0;
}

