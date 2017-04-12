// xim : basic framework to handle XIMEA cameras with RPIt socket block
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
//	For optimal reading, set tab size = 2 in your editor.
//
// JG, 30.03.17

//#define XIM_OPENCV_VER3										// Set this flag to compile with OpenCV 3.x

#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <signal.h>
#include <pthread.h>
#include <termios.h>
#include <memory.h>
#include <m3api/xiApi.h>
#include <iostream>
#ifdef XIM_OPENCV_VER3
#include <opencv2/core.hpp> 
#include <opencv2/features2d.hpp>
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#else
#include <opencv2/core/core.hpp> 
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif

// Socket handler definitions

#define RPIT_SOCKET_CON_N					10			// Nb of double sent (control)
#define RPIT_SOCKET_MES_N					10			// Nb of double returned (measurement)
#define RPIT_SOCKET_PORT					"31415"	// Port of the server
#define RPIT_SOCKET_MES_PERIOD		2000		// Sampling period of the measurement (us)
#define RPIT_SOCKET_MAGIC					3141592	// Magic number
#define RPIT_SOCKET_WATCHDOG_TRIG	1000000	// Delay in us before watchdog is triggered

struct RPIt_socket_mes_struct	{
	unsigned int				magic;							// Magic number
	unsigned long long 	timestamp;					// Absolute server time in ns 
	double							mes[RPIT_SOCKET_MES_N];	// Measurements
};

struct RPIt_socket_con_struct	{
	unsigned int				magic;							// Magic number
	unsigned long long 	timestamp;					// Absolute client time in ns
	double							con[RPIT_SOCKET_CON_N];	// Control signals
};

pthread_t 												mes_thread;
pthread_mutex_t 									mes_mutex;
struct RPIt_socket_mes_struct			mes;
struct RPIt_socket_con_struct			con;
unsigned char											exit_req = 0;

// Image processing definitions

#define HandleResult(res,place) if (res!=XI_OK) {flockfile(stderr);fprintf(stderr,"XIAPI: error after %s (%d).\n",place,res);funlockfile(stderr);rpit_socket_cleanup( EXIT_FAILURE );}

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
#define XIM_BLOB_MIN_CIRC		0.5
#define XIM_BLOB_MAX_CIRC		1.0
//#define XIM_BLOB_MIN_CONVEX	0.87
//#define XIM_BLOB_MAX_CONVEX	1.0
//#define XIM_BLOB_MIN_INER_R	0.01
//#define XIM_BLOB_MAX_INER_R	0.1

XI_IMG 												image;
HANDLE 												xiH = NULL;
XI_RETURN 										stat = XI_OK;
IplImage*											cv_image = NULL;

using namespace 							cv;
using namespace 							std;

// Cleanup code
void rpit_socket_cleanup( int exit_code )	{
	
	#ifdef XIM_LIVE_VIDEO
	destroyWindow( XIM_VIDEO_NAME );
	#endif
	
	if ( cv_image != NULL )
		cvReleaseImage(	&cv_image	);
		
	if ( xiH != NULL )	{
		xiStopAcquisition( xiH );
		xiCloseDevice( xiH );
	}
	
	exit( exit_code );
}

// Get system time
void rpit_socket_get_time( struct timespec *ts )	{

	if ( !ts )
		return;

	clock_gettime( CLOCK_MONOTONIC, ts );
}

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
			flockfile( stderr );
			fprintf( stderr, "XIAPI: unknown format in xim_xiimg2cvipl.\n" );
			funlockfile( stderr );
			return NULL;
		}
	
	// Defines the pointer to image data
	cvipl->imageData=(char*)xiimg->bp;

	return cvipl;
}

//
// MEASUREMENT THREAD. Runs asynchronously at a higher rate.
//
void *rpit_socket_server_update( void *ptr )	{
	
	struct timespec 								current_time, last_time, comp_time;
	unsigned long long							period, comput;
	unsigned long long							watchdog_counter = 0;
	unsigned long long							last_timestamp = 0;
	int															i;
	
	cv::Mat													cv_mat, cv_im_bin;
	#ifdef XIM_LIVE_VIDEO
	cv::Mat													im_with_keypoints;
	unsigned long long							frame_cnt = 0;
	#endif
	cv::SimpleBlobDetector::Params 	params;
	#ifdef XIM_OPENCV_VER3
	cv::Ptr<cv::SimpleBlobDetector> detector;
	#endif
	std::vector<KeyPoint> 					keypoints, keypoints_ROI;
	unsigned char										detected = 0;
	std::vector<Rect>								xim_ROI( XIM_NB_FEATURES );
	std::vector<Mat>								xim_ROI_mat( XIM_NB_FEATURES );
	std::vector<Point2f>						xim_cog( XIM_NB_FEATURES );
	
	
	rpit_socket_get_time( &last_time );
	mes.magic = RPIT_SOCKET_MAGIC;
	
	// OpenCV initialization
	
	// Get XIAPI pointer to image
	cv_image = xim_xiimg2cvipl( &image );
	if ( cv_image == NULL )	{
		flockfile( stderr );
		fprintf( stderr, "XIAPI: error after xim_xiimg2cvipl (NULL).\n" );
		funlockfile( stderr );
		rpit_socket_cleanup( EXIT_FAILURE );
		return NULL;
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
	#ifdef XIM_OPENCV_VER3
	detector = cv::SimpleBlobDetector::create( params );
	#else
	cv::SimpleBlobDetector detector( params );
	#endif
	
	while( 1 )	{
		
		/* Check if exit is requested */
		
		if ( exit_req )
			break;	
		
		/* 
		 * 
		 * 
		 * Insert the measurements acquisition code here.
		 * This code can used control signals safely:
		 * they are protected by a mutex.
		 * 
		 * 
		 * 
		 */
			
		// Getting image from camera
		stat = xiGetImage( xiH, XIM_TIMEOUT, &image );
		HandleResult( stat, "xiGetImage" );
		
		// Enter critical section
	
		pthread_mutex_lock( &mes_mutex );	

		// Get current time
		rpit_socket_get_time( &current_time );
		mes.timestamp = (unsigned long long)current_time.tv_sec * 1000000000
									+ (unsigned long long)current_time.tv_nsec;
		
		// Protect all OpenCV code with a try...catch
		
		try	{

			// Update CV image
			cv_image->imageData=(char*)image.bp;
			cv_mat = cv::cvarrToMat( cv_image );

			// Detect blobs only if needed (takes time)
			if ( !detected )
			{
				// Image binarization
				cv::threshold( cv_mat, cv_im_bin, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU );
				// Global blob detection
				#ifdef XIM_OPENCV_VER3
				detector->detect( cv_im_bin, keypoints );
				#else
				detector.detect( cv_im_bin, keypoints );
				#endif
				if ( keypoints.size() == XIM_NB_FEATURES )	{
					flockfile( stdout );
					printf( "OpenCV: all features detected.\n" );
					funlockfile( stdout );
					// Initializing ROIs
					for ( i = 0; i < XIM_NB_FEATURES; i++ )	{
						#ifndef XIM_OPENCV_VER3
						keypoints[i].size *= 2.0;
						#endif
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
				{
					flockfile( stderr );
					fprintf( stderr, "OpenCV: missing features.\n" );
					funlockfile( stderr );
				}
			}

			// If features are localized, switch to tracking mode
			if ( detected )
			{	
				flockfile( stdout );
				printf( "OpenCV: feature tracking mode.\n" );
				funlockfile( stdout );
				
				// Detect blob in ROIs
				for ( i = 0; i < XIM_NB_FEATURES; i++ )	{
					
					// Extract ROI from acquired image
					xim_ROI_mat[i] = cv_mat( xim_ROI[i] );
					
					// ROI binarization
					cv::threshold( xim_ROI_mat[i], cv_im_bin, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU );
					
					// Blob detection in ROI
					#ifdef XIM_OPENCV_VER3
					detector->detect( cv_im_bin, keypoints_ROI );
					#else
					detector.detect( cv_im_bin, keypoints_ROI );
					#endif
					
					// Error handling
					if ( keypoints_ROI.size() == 0 )	{
						flockfile( stderr );
						fprintf( stderr, "OpenCV: no feature in ROI #%d!\n", i );
						funlockfile( stderr );
						detected = 0;
						break;
					}
					if ( keypoints_ROI.size() > 1 )	{
						flockfile( stderr );
						fprintf( stderr, "OpenCV: more than 1 feature in ROI #%d!\n", i );
						funlockfile( stderr );
					}
					
					// Update keypoint
					keypoints[i] = keypoints_ROI[0];
					keypoints[i].pt.x += xim_ROI[i].x;
					keypoints[i].pt.y += xim_ROI[i].y;
					#ifndef XIM_OPENCV_VER3
					keypoints[i].size *= 2.0;
					#endif
					
					// Update ROI
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
					
					// Extract feature center of gravity coordinates
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
					rpit_socket_cleanup( EXIT_SUCCESS );
			}
			#else
			// Exit if key pressed
			if ( kbhit( ) )
				rpit_socket_cleanup( EXIT_SUCCESS );
			#endif
		}
		catch( cv::Exception& e )
		{
			const char* err_msg = e.what();
			
			flockfile( stderr );
			fprintf( stderr, "OpenCV: exception caught >> %s\n", err_msg );
			funlockfile( stderr );
			rpit_socket_cleanup( EXIT_FAILURE );
		}

		/**********************************************/

		/* Update measurements when features are detected */
		
		if ( detected )	{
			for( i = 0; ( i < XIM_NB_FEATURES ) && ( 2*i+1 < RPIT_SOCKET_MES_N ); i++ )	{
				mes.mes[2*i] = xim_cog[i].x;
				mes.mes[2*i+1] = xim_cog[i].y;
			}
		}

		/* Whatchdog: if control signals are not updated, force them to 0 */

		if ( last_timestamp != con.timestamp )	{
			watchdog_counter = 0;
			last_timestamp = con.timestamp;
		}
		else
			watchdog_counter++;
		
		if ( watchdog_counter >= ( RPIT_SOCKET_WATCHDOG_TRIG / RPIT_SOCKET_MES_PERIOD ) )	{
			
			flockfile( stderr );
			fprintf( stderr, "rpit_socket_server_update: watchdog triggered (%ds).\n",
												(int)( ( watchdog_counter * RPIT_SOCKET_MES_PERIOD ) / 1000000 ) );
			funlockfile( stderr );
			
			for( i = 0; i < RPIT_SOCKET_CON_N; i++ )
				con.con[i] = 0.0;
		}
	
		pthread_mutex_unlock( &mes_mutex );	
		
		/* Display timing stats */
		
		rpit_socket_get_time( &comp_time );
		
		comput =  ( (unsigned long long)comp_time.tv_sec * 1000000000
							+ (unsigned long long)comp_time.tv_nsec ) - mes.timestamp;
		
		period = mes.timestamp - ( (unsigned long long)last_time.tv_sec * 1000000000
														 + (unsigned long long)last_time.tv_nsec );
		last_time = current_time;
		
		flockfile( stdout );
		printf( "rpit_socket_server_update: period duration = %llu us.\n", period / 1000 );
		printf( "rpit_socket_server_update: cpu usage = %d percent.\n", (int)( (double)comput / (double)period * 100.0 ) );
		funlockfile( stdout );
	}
	
	return NULL;
}

// SIGINT handler : performs the cleanup
void rpit_socket_server_int_handler( int dummy )	{
	
	/* Request termination of the thread */
	
	exit_req = 1;
	
	/* Wait for thread to terminate */
	
	pthread_join( mes_thread, NULL );
	
	flockfile( stderr );
	fprintf( stderr, "\nrpit_socket_server_int_handler: measurement thread stopped. Cleaning up...\n" );
	funlockfile( stderr );
	
	/* Cleanup */
	
	rpit_socket_cleanup( EXIT_SUCCESS );

}

int main( int argc, char* argv[] )
{
	struct addrinfo 								hints;
	struct addrinfo 								*result, *rp;
	int 														sfd, s, i;
	struct sockaddr_storage 				peer_addr;
	socklen_t 											peer_addr_len;
	ssize_t 												nread;
	struct RPIt_socket_mes_struct		local_mes;
	struct RPIt_socket_con_struct		local_con;
	
	/* 
	 * 
	 * 
	 * XIAPI initialization
	 * 
	 * 
	 * 
	 */
	
	// Initialize image buffer
	memset( &image, 0, sizeof(image) );
	image.size = sizeof( XI_IMG );

	// Retrieving a handle to the camera device 
	printf( "XIAPI: Opening first camera.\n" );
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
	printf( "XIAPI: Starting acquisition.\n" );
	stat = xiStartAcquisition( xiH );
	HandleResult( stat, "xiStartAcquisition" );

	// Get first image to detect format
	stat = xiGetImage( xiH, XIM_TIMEOUT, &image );
	HandleResult( stat, "xiGetImage" );

	/**********************************************/
	
	/* Initialize mutex */
	
	pthread_mutex_init( &mes_mutex, NULL );
	
	/* Clear mes structure */
	
	mes.timestamp = 0;
	for ( i = 0; i < RPIT_SOCKET_MES_N; i++ )
		mes.mes[i] = 0.0;
	
	/* Clear con structure */
	
	con.magic = 0;
	con.timestamp = 0;
	for ( i = 0; i < RPIT_SOCKET_CON_N; i++ )
		con.con[i] = 0.0;
	
	/* Initialize SIGINT handler */
	
	signal( SIGINT, rpit_socket_server_int_handler );
	
	memset( &hints, 0, sizeof( struct addrinfo ) );
	hints.ai_family = AF_UNSPEC;    /* Allow IPv4 or IPv6 */
	hints.ai_socktype = SOCK_DGRAM; /* Datagram socket */
	hints.ai_flags = AI_PASSIVE;    /* For wildcard IP address */
	hints.ai_protocol = 0;					/* Any protocol */
	hints.ai_canonname = NULL;
	hints.ai_addr = NULL;
	hints.ai_next = NULL;

	s = getaddrinfo( NULL, RPIT_SOCKET_PORT, &hints, &result );
	
	if ( s != 0 ) {
		flockfile( stderr );
		fprintf( stderr, "rpit_socket_server: function getaddrinfo returned: %s\n", gai_strerror( s ) );
		funlockfile( stderr );
		rpit_socket_cleanup( EXIT_FAILURE );
	 }
	 
	/* 	
		getaddrinfo() returns a list of address structures.
		Try each address until we successfully bind(2).
		If socket(2) (or bind(2)) fails, we (close the socket
		and) try the next address. 
	*/

	for ( rp = result; rp != NULL; rp = rp->ai_next ) {
		sfd = socket( rp->ai_family, rp->ai_socktype, rp->ai_protocol );
		if ( sfd == -1 )
			continue;

		if ( bind( sfd, rp->ai_addr, rp->ai_addrlen ) == 0 )
			break;									/* Success */

		close( sfd );
	}

	if ( rp == NULL ) {					/* No address succeeded */
		flockfile( stderr );
		fprintf( stderr, "rpit_socket_server: could not bind. Aborting.\n" );
		funlockfile( stderr );
		exit( EXIT_FAILURE );
	}

	freeaddrinfo( result );			/* No longer needed */ 
	
	/* Start measurement thread */
	
	pthread_create( &mes_thread, NULL, rpit_socket_server_update, (void*) NULL );
	
	/* Wait for control datagram and answer measurement to sender */

	while ( 1 ) {
		
		/* Read control signals from the socket */
		
		peer_addr_len = sizeof( struct sockaddr_storage );
		nread = recvfrom(	sfd, (char*)&local_con, sizeof( struct RPIt_socket_con_struct ), 0,
											(struct sockaddr *)&peer_addr, &peer_addr_len );
		
		/* Memcopy is faster than socket read: avoid holding the mutex too long */
		
		pthread_mutex_lock( &mes_mutex );
		
		memcpy( &con, &local_con, sizeof( struct RPIt_socket_con_struct ) );
		
		if ( nread == -1 )	{
			flockfile( stderr );
			fprintf( stderr, "rpit_socket_server: function recvfrom exited with error.\n" );
			funlockfile( stderr );
			
			/* Clear control in case of error */
			
			for ( i = 0; i < RPIT_SOCKET_CON_N; i++ )
				con.con[i] = 0.0;
		}
		
		if ( nread != sizeof( struct RPIt_socket_con_struct ) )	{
			flockfile( stderr );
			fprintf( stderr, "rpit_socket_server: function recvfrom did not receive the expected packet size.\n" );
			funlockfile( stderr );
			
			/* Clear control in case of error */
			
			for ( i = 0; i < RPIT_SOCKET_CON_N; i++ )
				con.con[i] = 0.0;
		}
										
		if ( con.magic != RPIT_SOCKET_MAGIC )	{
			flockfile( stderr );
			fprintf( stderr, "rpit_socket_server: magic number problem. Expected %d but received %d.\n", RPIT_SOCKET_MAGIC, con.magic );
			funlockfile( stderr );
			
			/* Clear control in case of error */
			
			for ( i = 0; i < RPIT_SOCKET_CON_N; i++ )
				con.con[i] = 0.0;
		}

		pthread_mutex_unlock( &mes_mutex );
		
		/*
		 * 
		 * 
		 *	Insert here the handling of control signals.
		 * 
		 * 
		 * 
		 */
		 

		/**********************************************/
		
		/* Critical section : copy of the measurements to a local variable */
		
		pthread_mutex_lock( &mes_mutex );
		memcpy( &local_mes, &mes, sizeof( struct RPIt_socket_mes_struct ) );
		pthread_mutex_unlock( &mes_mutex );	
		
		/* Send measurements to the socket */
		
		if ( sendto(	sfd, (char*)&local_mes, sizeof( struct RPIt_socket_mes_struct ), 0,
									(struct sockaddr *)&peer_addr,
									peer_addr_len) != sizeof( struct RPIt_socket_mes_struct ) )	{
			flockfile( stderr );
			fprintf( stderr, "rpit_socket_server: error sending measurements.\n" );
			funlockfile( stderr );
		}
	}
		
	exit( EXIT_SUCCESS );
}

